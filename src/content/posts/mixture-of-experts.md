---
title: "671B인데 토큰 하나엔 37B만 쓴다"
subtitle: "Mixture-of-Experts — 파라미터 수와 연산량을 떼어놓는 라우팅의 수학"
description: "FFN을 여러 expert로 쪼개고 토큰마다 top-k만 켠다. 라우터 식과 연산량, load-balancing 보조 손실의 유도와 DeepSeek-V3의 손실 없는 균형법, 세분화 expert와 shared expert, 통신 비용까지 — MoE가 2025년 표준이 된 이유."
pubDate: 2026-09-03
order: 8
track: "아키텍처"
tags: ["MoE","Mixture-of-Experts","DeepSeek-V3","routing","아키텍처"]
hero: "/images/moe_routing.png"
---

scaling law는 "파라미터가 많을수록 loss가 낮다"고 말합니다. 그런데 dense 모델에서 파라미터를 10배 늘리면 **토큰 하나를 처리하는 연산도 10배** 듭니다. 학습도, 추론도요.

Mixture-of-Experts(MoE)는 이 연결을 끊습니다. DeepSeek-V3는 파라미터가 **671B**이지만 토큰 하나가 실제로 지나가는 파라미터는 **37B**입니다. Mixtral 8x7B는 46.7B 중 12.9B, gpt-oss-120b는 117B 중 5.1B. 이 글은 그 트릭 — 라우팅 — 을 수식으로 뜯어보고, 왜 어렵고(load balancing), 어떻게 풀었는지(DeepSeek의 loss-free 균형)까지 따라갑니다. 용어는 영어를 그대로 씁니다.

## 1. 어디를 쪼개나 — FFN

Transformer 층은 attention과 **FFN**(feed-forward network, MLP)으로 구성됩니다. 파라미터의 대부분(보통 2/3)은 FFN에 있습니다. 토큰 하나의 표현 $x\in\mathbb{R}^{d}$에 대해 FFN은

$$
\mathrm{FFN}(x) = W_2\,\sigma(W_1 x),\qquad W_1\in\mathbb{R}^{d_{ff}\times d},\ W_2\in\mathbb{R}^{d\times d_{ff}} ,
$$

$d$는 모델 차원, $d_{ff}$는 은닉 차원(보통 $4d$), $\sigma$는 비선형 함수(GELU, SwiGLU 등)입니다. 파라미터는 $2\,d\,d_{ff}$개이고, 토큰당 연산은 곱셈-덧셈을 2 FLOPs로 세어 약 $4\,d\,d_{ff}$ FLOPs입니다. 핵심 관찰: **FFN은 토큰마다 독립적으로** 적용됩니다(attention과 달리 다른 토큰을 보지 않음). 그래서 토큰마다 **다른 FFN을 골라 써도** 구조가 깨지지 않습니다.

## 2. MoE 층의 정의

FFN 하나 대신 **expert** $E$개 $\{\mathrm{FFN}_1,\dots,\mathrm{FFN}_E\}$를 두고, **router**가 토큰마다 어느 expert를 쓸지 정합니다.

$$
s = W_g\,x\in\mathbb{R}^{E},\qquad
g = \mathrm{softmax}(s),\qquad
\mathcal{K} = \mathrm{TopK}(g,\,k),\qquad
y = \sum_{e\in\mathcal{K}} \tilde g_e\,\mathrm{FFN}_e(x).
$$

기호를 봅시다. $W_g\in\mathbb{R}^{E\times d}$는 라우터의 가중치(작은 선형층 하나), $s$는 각 expert에 대한 **점수(logit)**, $g_e$는 expert $e$가 뽑힐 **확률(affinity)**, $\mathcal{K}$는 확률이 높은 순으로 고른 $k$개 expert의 집합, $\tilde g_e$는 뽑힌 $k$개 안에서 다시 합이 1이 되도록 정규화한 가중치($\tilde g_e = g_e/\sum_{e'\in\mathcal K}g_{e'}$; Mixtral 방식)입니다. 즉 **토큰은 $E$개 중 $k$개 expert만 통과**하고, 출력은 그 $k$개의 가중합입니다.

![top-k 라우팅과 활성/총 파라미터](/images/moe_routing.png)
> 왼쪽: $E{=}8,\ k{=}2$인 라우팅. 라우터가 확률 $g$를 내고 상위 2개(expert 2, 5)만 실행합니다. 나머지 6개는 이 토큰에 대해 연산하지 않습니다. 오른쪽: 공개 MoE 모델들의 총 파라미터 대 토큰당 활성 파라미터(로그축). 점선은 dense 모델(활성=총)입니다. 점선에서 아래로 멀수록 "큰 모델을 싸게 돌리는" 정도가 큽니다.

**파라미터와 연산의 분리**는 이 정의에서 바로 나옵니다.

$$
\text{총 파라미터} = E\cdot 2\,d\,d_{ff},\qquad
\text{토큰당 FLOPs} \approx k\cdot 4\,d\,d_{ff} + \underbrace{2\,E\,d}_{\text{router}} .
$$

$E$를 키우면 총 파라미터(=지식 용량)는 비례해 늘지만, 토큰당 연산은 $k$에만 비례합니다. 라우터 비용 $2Ed$는 $E$가 수백이어도 FFN 대비 무시할 만합니다. 쉽게 말해, **도서관은 크게 짓되 한 질문에 사서는 $k$명만 움직입니다.**

## 3. 라우팅은 왜 어려운가 — load balancing

문제는 라우터가 학습 중에 **몇 개 expert만 편애**하기 쉽다는 겁니다. 초기에 조금 더 잘하는 expert가 더 많이 뽑히고 → 더 많이 학습돼 더 잘하고 → 더 뽑히는 양의 되먹임이 생깁니다(routing collapse). 결과는 expert 대부분이 노는 채로 사실상 dense 소형 모델이 되는 것. 게다가 expert들이 GPU에 나눠 실려 있으면(expert parallelism), 몰린 expert가 있는 GPU만 바쁘고 나머지는 대기합니다.

**보조 손실(auxiliary loss)** 이 고전적 해법입니다(Switch Transformer, 2021). 배치의 토큰 수를 $T$라 하고,

$$
f_e = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\,e\in\mathcal{K}_t\,],\qquad
P_e = \frac{1}{T}\sum_{t=1}^{T} g_{e}(x_t),\qquad
\mathcal{L}_{\text{aux}} = \alpha\,E\sum_{e=1}^{E} f_e\,P_e .
$$

$f_e$는 expert $e$로 **실제로 라우팅된 토큰의 비율**(불연속이라 미분 불가), $P_e$는 expert $e$에 배정된 **평균 확률**(미분 가능), $\alpha$는 손실의 세기입니다. 왜 이 곱이 균형을 유도할까요? $f$와 $P$는 같은 라우터에서 나오니 대략 같은 분포를 따릅니다. $f_e\approx P_e$로 놓으면 $\sum_e f_eP_e\approx\sum_e P_e^2$이고, $\sum_e P_e=1$ 제약 아래 Cauchy–Schwarz에 의해

$$
\sum_{e=1}^{E}P_e^{2}\;\ge\;\frac{1}{E}\Big(\sum_e P_e\Big)^2=\frac{1}{E},
$$

등호는 **모든 $P_e=1/E$일 때** 성립합니다. 즉 $\mathcal L_{\text{aux}}$의 최솟값 $\alpha$는 균등 배분에서 달성되고, 편중될수록 커집니다. 미분은 $P_e$를 통해 흐르므로 라우터가 **덜 뽑힌 expert의 확률을 올리는 방향**으로 학습됩니다.

대가가 있습니다. $\alpha$가 크면 라우터가 "이 토큰에 맞는 expert"보다 "균형"을 우선해 **품질이 떨어지고**, 작으면 균형이 안 잡힙니다. 이 줄타기가 MoE 학습의 고질적 난점이었습니다.

## 4. DeepSeek-V3 — 손실 없이 균형 잡기 (auxiliary-loss-free)

DeepSeek-V3(2024)는 보조 손실을 **없애고**, 대신 expert마다 **bias** $b_e$를 두어 **선택에만** 개입합니다.

$$
\mathcal{K}_t = \mathrm{TopK}\big(\{\,s_{t,e}+b_e\,\}_{e=1}^{E},\ k\big),\qquad
y_t=\sum_{e\in\mathcal K_t}\tilde g_{t,e}\,\mathrm{FFN}_e(x_t)\quad(\tilde g\text{는 }b\text{ 없이 }s\text{로 계산}).
$$

$s_{t,e}$는 토큰 $t$의 expert $e$ 점수(V3는 softmax 대신 sigmoid affinity), $b_e$는 expert $e$의 **선택 편향**입니다. 학습 스텝마다 각 expert의 부하를 보고, **과부하면 $b_e\mathrel{-}=\gamma$, 저부하면 $b_e\mathrel{+}=\gamma$** 로 조정합니다($\gamma$는 작은 상수, 예: 0.001). 핵심은 $b_e$가 **어느 expert를 고를지에만** 영향을 주고, **가중치 $\tilde g$에는 영향을 주지 않는다**는 점입니다. 그래서 gradient에는 아무 왜곡이 없고, 균형은 제어기(controller)처럼 바깥에서 맞춥니다. 논문은 이 방식이 보조 손실 방식보다 같은 균형에서 더 낮은 loss를 낸다고 보고합니다(arXiv:2408.15664). 시퀀스 단위 극단 편중을 막기 위한 아주 작은 보조 손실은 남겨 둡니다.

## 5. Expert를 더 잘게, 그리고 공유 expert — DeepSeekMoE

DeepSeek 계열의 또 다른 두 설계.

**세분화(fine-grained) expert.** $d_{ff}$를 $m$분의 1로 줄인 작은 expert를 $mE$개 두고 $mk$개를 고르면, 연산·파라미터는 그대로인데 **조합의 수가 폭증**합니다. $E=16,\ k=2$면 $\binom{16}{2}=120$가지, $m=4$로 쪼개면 $\binom{64}{8}\approx 4.4\times10^9$가지. 같은 예산으로 훨씬 다양한 "전문가 조합"을 만들 수 있습니다. V3는 라우팅 expert **256개 중 8개**를 씁니다.

**공유(shared) expert.** 모든 토큰이 **항상** 통과하는 expert $n_s$개를 따로 둡니다.

$$
y = \sum_{i=1}^{n_s}\mathrm{FFN}^{(s)}_i(x) + \sum_{e\in\mathcal K}\tilde g_e\,\mathrm{FFN}_e(x).
$$

공통 지식(문법, 흔한 패턴)을 공유 expert가 맡으면, 라우팅 expert들이 그걸 **중복 학습할 필요가 없어져** 더 전문화됩니다. V3는 $n_s=1$입니다.

## 6. 시스템 비용 — 통신과 메모리

MoE의 비용은 FLOPs가 아니라 **다른 곳**에서 나옵니다.

- **메모리.** 활성은 37B여도 **671B 전부가 GPU 메모리에 상주**해야 합니다(어느 expert가 뽑힐지 모르니). bf16이면 1.3TB — 노드 하나로는 어림없어 여러 노드에 expert를 분산(expert parallelism)합니다.
- **통신.** 토큰이 자기 expert가 있는 GPU로 **이동**해야 합니다. MoE 층마다 all-to-all 통신이 두 번(보내기, 결과 받기) 일어납니다. V3는 토큰 하나가 최대 4개 노드까지만 가도록 제한(node-limited routing)하고, 통신을 연산과 겹치는 파이프라인(DualPipe)으로 숨깁니다.
- **추론 배치.** 배치가 작으면 뽑힌 expert만 드문드문 실행돼 GPU 활용률이 낮습니다. 대규모 서빙에서는 배치를 키워 expert마다 토큰이 충분히 모이게 하고, expert를 GPU에 분산해 병렬로 돌립니다. **MoE는 대규모 배치 서빙에서 가장 경제적**이고, 단일 사용자 로컬 추론에서는 이점이 줄어듭니다.

## 7. 왜 loss가 낮아지나 — 그리고 한계

같은 학습 연산에서 MoE가 dense보다 loss가 낮은 것은 반복 확인된 사실입니다(Switch: 같은 FLOPs로 T5 대비 최대 7배 빠른 수렴). 직관은 §2의 식 그대로 — **연산은 그대로 두고 지식 용량만 키웠으니** 더 많은 사실·패턴을 저장할 수 있습니다. 다만 몇 가지 한계가 남습니다.

- **활성 파라미터당 능력**은 dense보다 낮습니다. 37B 활성의 V3가 dense 37B보다는 훨씬 낫지만, dense 671B만큼은 아닙니다. 추론(reasoning)처럼 깊은 연산이 필요한 과제에서 이 격차가 두드러진다는 관찰이 있습니다.
- **fine-tuning 불안정.** 라우팅이 소량 데이터에서 쉽게 흔들려 SFT에서 dense보다 과적합·붕괴가 잦습니다.
- **해석.** expert가 "수학 expert, 코드 expert"처럼 사람 눈에 보이는 분업을 하지는 않습니다. 대개 토큰 수준의 통계적 분업입니다.

## 8. 결론

MoE는 "**연산은 $k$개, 지식은 $E$개**"라는 한 줄의 분리입니다. 그 대가로 라우팅의 균형(§3–4)과 시스템의 통신·메모리(§6)라는 새 문제를 떠안았고, DeepSeek-V3가 손실 없는 균형·세분화·공유 expert·통신 숨김으로 그 문제들을 한꺼번에 풀어 보이면서 2025년 이후 대형 모델의 기본형이 됐습니다. 지난 편의 hybrid가 시퀀스 축의 비용을, MoE가 채널 축의 비용을 떼어냈으니 — 다음은 attention 자체의 KV cache를 줄이는 차례입니다.

*(참고: Shazeer et al. 2017 — arXiv:1701.06538 · Switch Transformer — arXiv:2101.03961 · Mixtral — arXiv:2401.04088 · DeepSeekMoE — arXiv:2401.06066 · auxiliary-loss-free balancing — arXiv:2408.15664 · DeepSeek-V3 — arXiv:2412.19437. 그림의 모델 수치는 각 공개 자료 기준이며, gpt-oss·Qwen3·Llama 4는 2025년 공개로 원문 재확인을 권합니다.)*

#MoE #MixtureOfExperts #DeepSeekV3 #routing #아키텍처
