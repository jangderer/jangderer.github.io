---
title: "정답만 채점해도 추론이 자란다"
subtitle: "RLVR과 GRPO로 보는 DeepSeek-R1 — 사람 선호 대신 검증 가능한 보상으로, critic 없이"
description: "검증 가능한 보상(RLVR)과 critic을 그룹으로 대체한 GRPO로, SFT 없이 RL만으로 긴 CoT가 창발한 DeepSeek-R1의 학습법을 유도까지 따라간다."
pubDate: 2026-09-02
order: 5
track: "정렬·추론"
tags: ["RLVR","GRPO","DeepSeek-R1","reasoning","정렬"]
hero: "/images/grpo_advantage.png"
---

지난 정렬 편들에서 모델은 **사람의 선호**로 정렬됐습니다(RLHF, DPO). DeepSeek-R1은 다른 길을 갑니다 — 사람 선호 대신 **검증 가능한 보상**(정답 여부)만으로 강화학습을 돌렸더니, **추론 능력이 스스로 자라났습니다.** 이 글에서는 그 엔진인 **RLVR**과 **GRPO**를 유도하고, R1이 실제로 어떻게 학습됐는지를 봅니다. 용어는 영어를 그대로 씁니다.

## 1. RLVR — 검증 가능한 보상

RLVR(RL with Verifiable Rewards)의 보상은 신경망(reward model)이 아니라 **규칙**입니다. 수학 문제면 정답과 대조하고, 코드면 테스트 통과 여부를 봅니다.

$$
r(o) = \begin{cases} 1 & \text{정답 / 테스트 통과} \\ 0 & \text{그 외} \end{cases}\quad (+\ \text{형식 보상})
$$

사람이 라벨한 선호도, 학습된 보상 모델도 없습니다. 그래서 학습된 보상의 허점을 악용하는 **reward hacking** 위험이 크게 줄고, 신호 자체가 정확합니다. 단, 자동 채점이 가능한 도메인(수학·코드·논리)에 한정됩니다.

## 2. 배경: policy gradient와 PPO의 무게

RL 목표는 기대 보상의 최대화입니다. 여기서 $\pi_\theta$는 파라미터 $\theta$를 가진 **정책**(응답을 생성하는 LLM 자신), $o\sim\pi_\theta$는 그 정책이 실제로 뽑은 **응답 하나**(토큰열), $r(o)$는 §1에서 정의한 그 응답의 검증 보상입니다.

$$
\max_\theta\ \mathbb{E}_{o\sim\pi_\theta}\big[\,r(o)\,\big],\qquad
\nabla_\theta \mathbb{E}[r] = \mathbb{E}\big[\,r\,\nabla_\theta\log\pi_\theta(o)\,\big].
$$

이 gradient는 분산이 큽니다. 그래서 baseline $b$를 빼서 **advantage** $A = r-b$로 바꿔 분산을 줄이죠. PPO는 여기에 (a) 중요도 비율을 클리핑해 안정화하고, (b) baseline을 대주는 **value network**(critic)를 따로 학습합니다. 문제는 critic이 **정책과 맞먹는 또 하나의 대형 모델**이라, 메모리와 연산이 배로 든다는 것입니다.

## 3. GRPO — critic을 그룹으로 대체 (유도)

GRPO(Group Relative Policy Optimization)의 아이디어는 하나입니다: **critic을 없애고, baseline을 그룹에서 얻자.**

한 프롬프트 $q$에 대해 정책이 $G$개 응답 $\{o_1,\dots,o_G\}$을 샘플링하고, 각각 검증 보상 $\{r_1,\dots,r_G\}$을 받습니다. baseline을 그 **그룹의 평균**으로 두고, advantage를 그룹 안에서 정규화합니다.

$$
\hat A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}.
$$

정답이 그룹 평균보다 나으면 양의 advantage, 못하면 음. 즉 "이 그룹 안에서 상대적으로 잘했나"로 정책을 밀고 당깁니다. **value network가 통째로 사라지죠.**

![GRPO는 critic 없이 그룹 내 상대 우위로 baseline을 만든다](/images/grpo_advantage.png)
> 한 프롬프트에 $G{=}8$개 응답을 뽑아 검증 보상(정답=1)을 매기고(왼쪽), 그룹 평균을 baseline으로 삼아 $\hat A_i=(r_i-\mathrm{mean})/\mathrm{std}$로 정규화합니다(오른쪽). 평균 이상이면 양(초록), 이하면 음(빨강). 별도 critic이 필요 없습니다.

목적함수는 PPO의 클리핑 형태를 그대로 쓰되 $\hat A_i$를 넣고, reference 정책과의 KL로 이탈을 제약합니다(개념 형태).

$$
\mathcal{J}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i}\min\big(\rho_i \hat A_i,\ \mathrm{clip}(\rho_i,\,1\!-\!\epsilon,\,1\!+\!\epsilon)\,\hat A_i\big)\right] - \beta\,\mathrm{KL}\!\big(\pi_\theta \Vert \pi_{\text{ref}}\big),
$$

여기서 각 기호는 이렇습니다. $\pi_\theta$는 **지금 학습 중인 현재 정책**, $\pi_{\theta_{\text{old}}}$는 이 배치의 응답들을 **뽑을 때 쓴 직전 정책**, $\pi_{\text{ref}}$는 학습 내내 **고정된 기준 정책**(보통 초기 SFT 모델)입니다. $\rho_i = \pi_\theta(o_i)/\pi_{\theta_{\text{old}}}(o_i)$는 두 정책이 응답 $o_i$에 부여하는 확률의 비(**중요도 비율**)로, 정책이 얼마나 변했는지를 재죠. $\mathrm{clip}(\rho_i,1{-}\epsilon,1{+}\epsilon)$의 $\epsilon$은 이 비율을 $[1{-}\epsilon,\,1{+}\epsilon]$ 범위로 자르는 **클리핑 폭**(한 번에 과도하게 업데이트되는 걸 막음), $\beta$는 KL 항의 세기를 정하는 **계수**입니다. critic이 없으니 메모리는 대략 절반, 파이프라인은 훨씬 단순해집니다.

## 4. R1-Zero — SFT 없이 RL만으로 추론이 창발

DeepSeek-R1-Zero의 결과는 충격적입니다. **base 모델에 SFT를 전혀 하지 않고 GRPO(RLVR)만** 돌렸는데,

- 응답이 **스스로 길어지고**(더 오래 "생각"),
- **self-verification·재검토** 같은 행동이 나타나며(이른바 "aha moment"),
- AIME 같은 수학 벤치마크 점수가 크게 오릅니다.

누구도 "길게 추론하라"고 가르치지 않았는데, **정답 보상만으로 긴 chain-of-thought가 창발**한 것이죠. 다만 R1-Zero는 가독성이 낮고 언어가 섞이는 약점이 있었습니다.

## 5. R1 — 실전 파이프라인

R1은 R1-Zero의 창발을 다듬어 실용화합니다.

```
① Cold-start SFT   소량의 고품질 long-CoT로 base를 살짝 정렬 (가독성 확보)
        ↓
② RLVR (GRPO)      검증 보상으로 추론 능력을 밀어올림
        ↓
③ Rejection sampling  RL 정책으로 좋은 응답을 뽑아 새 SFT 데이터 생성
        ↓
④ 최종 SFT + RL    추론·범용·안전을 함께 정렬
        ↓
   Distillation     큰 모델의 CoT를 작은 모델에 이식
```

## 6. 왜 중요한가

- **DPO 계열과의 대비**: DPO가 선호쌍의 closed-form 정렬이라면, RLVR/GRPO는 **정답이라는 객관 신호**로 RL을 돕니다. 최근엔 "GRPO는 사실 DPO와 통한다"는 분석도 나옵니다.
- **새 scaling 축**: pretraining보다 **RL·추론 compute**를 키우는 방향(post-training scaling)의 대표 엔진이 이 계열입니다.
- 그리고 이건 다음 편의 **test-time compute**(추론 시 연산을 더 써 품질을 얻기)와 짝을 이룹니다 — speculative decoding이 "값싸게 빨리"였다면, 이쪽은 "더 생각해서 더 정확히"입니다.

## 7. 결론

RLHF가 모델을 **사람 흉내 내도록** 정렬했다면, RLVR·GRPO는 모델이 **정답을 향해 스스로 추론을 조직**하도록 만듭니다. critic을 그룹으로 대체한 한 번의 단순화가, "RL만으로 추론이 창발한다"는 2025년의 전환을 열었습니다.

*(주의: R1 관련 수치·세부는 DeepSeek-R1 기술보고서(arXiv:2501.12948) 기준이며, 최신 후속 연구로 계속 갱신되고 있습니다.)*

#RLVR #GRPO #DeepSeekR1 #reasoning #정렬
