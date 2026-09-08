---
title: "학생이 직접 쓰고, 선생은 토큰마다 채점한다"
subtitle: "On-Policy Distillation — reverse KL을 per-token reward로 바꾸는 수학"
description: "SFT는 선생의 글을 베끼게 하고 RL은 답이 맞았는지만 알려준다. On-policy distillation은 학생이 스스로 쓴 문장의 토큰 하나하나에 선생의 log-prob 차이를 보상으로 준다. forward/reverse KL의 차이, 시퀀스 KL의 토큰 분해, policy gradient 유도, 실제 loss 구현과 GKD·MiniLLM·Thinking Machines 변형까지 수식으로 정리한다."
pubDate: 2026-09-08
order: 11
track: "정렬·추론"
tags: ["on-policy distillation","distillation","reverse KL","RL","정렬·추론"]
hero: "/images/opd_kl.png"
---

큰 모델이 아는 것을 작은 모델에 옮기는 방법은 크게 두 가지였습니다. 하나는 **선생 모델이 쓴 답을 학생이 그대로 따라 쓰게** 하는 것 — SFT, 흔히 "distillation"이라 부르는 그것입니다. 다른 하나는 **학생이 직접 풀게 하고 정답 여부로 보상**을 주는 것 — RLVR·GRPO 편에서 본 강화학습입니다.

둘 다 약점이 분명합니다. 앞의 것은 학생이 **자기가 쓴 적 없는 문맥**에서만 배웁니다. 시험장에서는 자기 손으로 쓴 앞 문장에 이어 써야 하는데, 연습은 늘 선생의 앞 문장 뒤에서만 했으니까요. 뒤의 것은 수천 토큰을 쓰고 나서 **"맞았다/틀렸다" 한 비트**만 돌아옵니다. 어느 토큰에서 잘못됐는지는 알 길이 없습니다.

**On-policy distillation(OPD)** 은 이 둘을 합칩니다. 학생이 **직접 씁니다**(on-policy). 그리고 선생은 그 글의 **토큰 하나하나에** "나라면 이 토큰을 얼마나 쓸 법했나"를 점수로 매깁니다(distillation). 2023년 GKD·MiniLLM이 형식을 잡았고, 2025년 Qwen3가 학습 파이프라인에 넣었으며, 같은 해 Thinking Machines의 블로그가 이름과 레시피를 대중화했습니다. 이 글은 그 loss가 **정확히 무엇이고 왜 그 형태인지**를 유도합니다. 용어는 영어를 그대로 씁니다.

## 1. 설정과 기호

프롬프트를 $x$, 응답을 토큰열 $y=(y_1,\dots,y_L)$이라 합시다. $L$은 응답 길이, 각 $y_t$는 어휘 $\mathcal{V}$의 원소입니다. 학생 모델은 파라미터 $\theta$를 가진 정책 $\pi_\theta$, 선생 모델은 고정된 정책 $\pi_T$입니다. 두 모델 모두 autoregressive라 시퀀스 확률은 토큰 조건부 확률의 곱입니다.

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{L}\pi_\theta(y_t\mid x,y_{<t}),\qquad
\pi_T(y\mid x)=\prod_{t=1}^{L}\pi_T(y_t\mid x,y_{<t}).
$$

$y_{<t}=(y_1,\dots,y_{t-1})$는 $t$번째 토큰 앞까지의 **prefix**입니다. 이하 표기를 줄여 $\pi_\theta(y_t\mid y_{<t})$처럼 $x$를 생략하겠습니다. 모든 확률은 $x$에 조건부라고 생각하시면 됩니다.

두 분포의 거리는 **Kullback–Leibler divergence**로 잽니다. 분포 $p,q$에 대해

$$
\mathrm{KL}(p\,\|\,q)=\mathbb{E}_{y\sim p}\!\left[\log\frac{p(y)}{q(y)}\right]=\sum_y p(y)\log\frac{p(y)}{q(y)}\;\ge 0,
$$

등호는 $p=q$일 때만입니다. **비대칭**이라는 점이 이 글의 절반입니다 — $\mathrm{KL}(p\|q)\ne\mathrm{KL}(q\|p)$. 어느 분포에서 샘플을 뽑아 기대값을 내느냐가 다르기 때문입니다.

## 2. 기존 두 방법의 loss — 무엇이 부족한가

### 2.1 SFT(off-policy distillation) = forward KL

선생이 생성한 응답 $y\sim\pi_T$를 모아 학생에게 cross-entropy로 학습시키는 것이 SFT입니다.

$$
\mathcal{L}_{\text{SFT}}(\theta)=\mathbb{E}_{y\sim\pi_T}\big[-\log\pi_\theta(y)\big]
=\underbrace{\mathbb{E}_{y\sim\pi_T}\!\left[\log\frac{\pi_T(y)}{\pi_\theta(y)}\right]}_{\mathrm{KL}(\pi_T\,\|\,\pi_\theta)}+\underbrace{\mathbb{E}_{y\sim\pi_T}[-\log\pi_T(y)]}_{H(\pi_T),\ \theta\text{와 무관}} .
$$

두 번째 항은 선생의 엔트로피라 상수이므로, **SFT는 forward KL $\mathrm{KL}(\pi_T\|\pi_\theta)$의 최소화**입니다. 기대값을 **선생의 분포**에서 잡습니다. 그래서 이름이 off-policy — 학습 데이터가 학생 자신의 정책에서 나오지 않았습니다.

문제는 두 가지입니다.

**(a) 분포 불일치(exposure bias).** 학습 때 학생은 항상 선생의 prefix $y_{<t}\sim\pi_T$ 뒤에서 다음 토큰을 예측합니다. 그런데 추론 때는 **자기가 만든** prefix $y_{<t}\sim\pi_\theta$ 뒤에 씁니다. 자기 실수가 섞인 문맥은 한 번도 본 적이 없으니, 한 토큰이 틀리면 그 뒤가 연쇄로 무너집니다. Ross & Bagnell(2010)의 imitation learning 분석이 이걸 정량화합니다 — 토큰당 오류율이 $\epsilon$이면 시퀀스 전체의 오류는 off-policy 학습에서 $O(\epsilon L^2)$, on-policy(자기 분포에서 학습)면 $O(\epsilon L)$까지 떨어집니다. 길이가 수천 토큰인 reasoning 응답에서 $L$ 대 $L^2$는 결정적 차이입니다.

**(b) mode-covering.** forward KL은 $\pi_T(y)>0$인 모든 $y$에서 $\pi_\theta(y)$가 너무 작으면 $\log(\pi_T/\pi_\theta)\to\infty$로 폭발합니다. 그래서 학생은 선생이 조금이라도 확률을 두는 **모든 곳을 덮어야** 합니다. 용량이 작은 학생은 결국 여러 mode 사이의 골짜기에도 확률을 흘리고, 거기서 샘플하면 어느 mode에도 속하지 않는 어정쩡한 답이 나옵니다.

### 2.2 RL = 시퀀스 단위 sparse reward

강화학습은 기대 보상을 최대화합니다. $R(x,y)$는 정답 여부 같은 스칼라 보상입니다.

$$
J(\theta)=\mathbb{E}_{y\sim\pi_\theta}\big[R(x,y)\big],\qquad
\nabla_\theta J=\mathbb{E}_{y\sim\pi_\theta}\!\Big[\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})\cdot A_t\Big].
$$

이것이 policy gradient이고, $A_t$는 토큰 $t$의 **advantage** — "이 토큰을 고른 것이 평균보다 얼마나 좋았나"입니다. GRPO는 $A_t$를 같은 프롬프트의 응답 $G$개 사이 정규화된 보상으로 두었죠(R1 편). 기대값이 **학생 자신의 분포**에서 잡히므로 (a)의 분포 불일치는 없습니다. 대신 신호가 **너무 성깁니다.** 응답 하나에 스칼라 하나 — 정보량으로 치면 $O(1)$ 비트가 $L$개 토큰 전부에 같은 값으로 뿌려집니다. 어느 토큰이 실수였는지는 수백 번의 샘플을 평균 내야 겨우 드러납니다. 게다가 검증 가능한 보상(verifier)이 있는 과제에서만 씁니다.

두 방법의 좋은 점만 남기면 이렇게 됩니다: **학생 분포에서 샘플하되, 토큰마다 밀도 있는 신호를.** 그 신호를 선생이 줍니다.

## 3. OPD의 loss — reverse KL

### 3.1 정의

OPD는 KL의 방향을 뒤집습니다.

$$
\mathcal{L}_{\text{OPD}}(\theta)=\mathrm{KL}(\pi_\theta\,\|\,\pi_T)
=\mathbb{E}_{y\sim\pi_\theta}\!\left[\log\frac{\pi_\theta(y)}{\pi_T(y)}\right].
$$

기대값을 **학생 분포**에서 잡습니다 — 학생이 직접 쓴 응답에 대해, "학생이 이 응답에 둔 확률"과 "선생이 이 응답에 둔 확률"의 log 비를 잽니다. 이것을 **reverse KL**이라 부릅니다.

### 3.2 시퀀스 KL의 토큰 분해 — 핵심 유도

시퀀스 확률이 곱이니 log 비는 합으로 풀립니다.

$$
\log\frac{\pi_\theta(y)}{\pi_T(y)}=\sum_{t=1}^{L}\Big[\log\pi_\theta(y_t\mid y_{<t})-\log\pi_T(y_t\mid y_{<t})\Big]
=\sum_{t=1}^{L}\rho_t(y),
$$

$\rho_t(y)$는 토큰 $t$에서의 **per-token log-ratio**입니다. 기대값을 취하면

$$
\mathrm{KL}(\pi_\theta\|\pi_T)=\mathbb{E}_{y\sim\pi_\theta}\Big[\sum_{t=1}^{L}\rho_t(y)\Big]
=\sum_{t=1}^{L}\mathbb{E}_{y_{<t}\sim\pi_\theta}\Big[\underbrace{\mathbb{E}_{y_t\sim\pi_\theta(\cdot\mid y_{<t})}\big[\rho_t\big]}_{=\ \mathrm{KL}\big(\pi_\theta(\cdot\mid y_{<t})\,\|\,\pi_T(\cdot\mid y_{<t})\big)}\Big].
$$

가운데 등식은 기대값을 prefix와 그 다음 토큰으로 나눈 것(tower property)입니다 — $y$ 전체에 대한 기대는 "먼저 $y_{<t}$를 뽑고, 그 조건에서 $y_t$를 뽑는" 두 단계와 같습니다. 안쪽 기대값은 정의 그대로 **그 위치의 조건부 분포 사이 KL**입니다. 결국

$$
\boxed{\;\mathrm{KL}(\pi_\theta\|\pi_T)=\sum_{t=1}^{L}\ \mathbb{E}_{y_{<t}\sim\pi_\theta}\Big[\mathrm{KL}_t(y_{<t})\Big],\qquad
\mathrm{KL}_t(y_{<t})\equiv\mathrm{KL}\big(\pi_\theta(\cdot\mid y_{<t})\,\|\,\pi_T(\cdot\mid y_{<t})\big)\;}
$$

시퀀스 하나의 큰 KL이 **"학생이 만든 prefix 위에서 잰 토큰별 KL"의 합**으로 정확히 쪼개집니다. 이것이 OPD의 모든 것입니다 — 신호가 토큰 단위로 **자연스럽게** 분해되고(RL처럼 억지로 credit을 나눌 필요가 없고), 그 신호가 잡히는 문맥은 **학생 자신의** prefix입니다(SFT의 분포 불일치가 없습니다). 참고로 forward KL도 같은 방식으로 쪼개지지만 prefix가 $\pi_T$에서 나옵니다. 방향 하나가 "누구의 문맥에서 배우느냐"를 정합니다.

### 3.3 왜 reverse인가 — mode-seeking과 zero-forcing

reverse KL의 피적분 $\log(\pi_\theta/\pi_T)$를 보면, 학생이 확률을 두는 곳($\pi_\theta>0$)에서 선생 확률이 0에 가까우면 $-\log\pi_T\to\infty$로 **강하게 벌**합니다. 반대로 선생이 확률을 두는 곳을 학생이 비워 두는 것은 — 거기서 샘플이 안 나오니 — **벌하지 않습니다.** 두 성질을 각각 **zero-forcing**("선생이 안 쓸 토큰은 절대 쓰지 마라")과 **mode-seeking**("선생의 여러 답 중 하나만 잘 하면 된다")이라 부릅니다.

![forward vs reverse KL, dense vs sparse credit](/images/opd_kl.png)
> 왼쪽: 두 봉우리를 가진 선생 분포에 단봉 Gaussian 학생을 맞춘 1차원 예시. forward KL 최소해(주황)는 두 봉우리를 모두 덮으려다 골짜기까지 퍼지고, reverse KL 최소해(파랑)는 한 봉우리에 정확히 올라타며 선생 확률이 0인 곳에는 확률을 두지 않습니다. 용량이 작은 학생에게 맞는 목표는 후자입니다. 오른쪽: 한 응답에서 토큰별로 떨어지는 OPD의 advantage $A_t=\log\pi_T(y_t)-\log\pi_\theta(y_t)$(파랑, 위치마다 하나씩 $L$개)와 RL의 terminal reward(주황, 응답당 하나). 크게 음수인 위치가 "선생이라면 쓰지 않았을 토큰"입니다. 그림은 직접 그린 것이며 수치는 예시입니다.

용량 논증을 붙이면 이렇습니다. 학생이 선생보다 작으면 $\pi_T$를 완전히 재현할 수 없습니다. 그때 forward KL은 "다 조금씩" 흉내 내는 평균적 학생을 만들고, reverse KL은 "일부는 완벽히" 흉내 내는 학생을 만듭니다. 생성 모델에서 우리가 원하는 것은 후자입니다 — 샘플 하나하나가 **선생이 썼을 법한** 글이어야지, 선생의 모든 가능성을 흐릿하게 섞은 글이면 안 됩니다.

### 3.4 gradient — 두 가지 estimator

이제 $\nabla_\theta\mathcal{L}_{\text{OPD}}$를 구합니다. 기대값의 분포 자체가 $\theta$에 의존하므로 REINFORCE의 score-function 항등식이 필요합니다. 일반적으로 $\theta$에 의존하는 함수 $f_\theta$에 대해

$$
\nabla_\theta\,\mathbb{E}_{y\sim\pi_\theta}[f_\theta(y)]
=\mathbb{E}_{y\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(y)\cdot f_\theta(y)\big]+\mathbb{E}_{y\sim\pi_\theta}\big[\nabla_\theta f_\theta(y)\big].
$$

첫 항은 "샘플링 분포가 움직여서" 생기는 변화, 둘째 항은 "피적분 함수가 움직여서" 생기는 변화입니다. 우리의 $f_\theta(y)=\log\pi_\theta(y)-\log\pi_T(y)$이므로 $\nabla_\theta f_\theta=\nabla_\theta\log\pi_\theta(y)$이고, 둘째 항은

$$
\mathbb{E}_{y\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(y)\big]=\sum_y \pi_\theta(y)\frac{\nabla_\theta\pi_\theta(y)}{\pi_\theta(y)}=\nabla_\theta\sum_y\pi_\theta(y)=\nabla_\theta 1=0 .
$$

score function의 기대값이 0이라는 고전적 사실입니다. 그러니 남는 것은 첫 항뿐이고, $\log\pi_\theta(y)=\sum_t\log\pi_\theta(y_t\mid y_{<t})$와 $f_\theta=\sum_s\rho_s$를 넣으면

$$
\nabla_\theta\mathrm{KL}(\pi_\theta\|\pi_T)
=\mathbb{E}_{y\sim\pi_\theta}\Big[\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})\cdot\sum_{s=1}^{L}\rho_s(y)\Big].
$$

여기서 인과성을 씁니다. 토큰 $t$의 선택은 그보다 **앞** 위치의 log-ratio $\rho_s\ (s<t)$에 영향을 줄 수 없으므로, $s<t$ 항은 기대값에서 사라집니다(조건부 기대를 취하면 $\mathbb{E}_{y_t}[\nabla\log\pi_\theta(y_t\mid y_{<t})]=0$이 곱해지기 때문 — 위와 같은 논리입니다). 따라서

$$
\nabla_\theta\mathrm{KL}(\pi_\theta\|\pi_T)
=\mathbb{E}_{y\sim\pi_\theta}\Big[\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})\cdot\underbrace{\sum_{s\ge t}\rho_s(y)}_{\text{reward-to-go}}\Big].
$$

§2.2의 policy gradient와 **완전히 같은 꼴**입니다. 보상이 $-\rho_s$(즉 $\log\pi_T-\log\pi_\theta$, 선생이 그 토큰을 학생보다 더 좋아할수록 양수), advantage가 그 reward-to-go인 RL입니다. reward model도, verifier도, value network도 없이 **선생의 log-prob 하나로 dense reward가 정의**됐습니다.

이제 두 갈래로 나뉩니다.

**estimator A — 위치별 exact KL.** reward-to-go 안에서 $s=t$인 항만 따로 보면, $y_{<t}$가 주어졌을 때

$$
\mathbb{E}_{y_t\sim\pi_\theta(\cdot\mid y_{<t})}\big[\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})\cdot\rho_t\big]
=\nabla_\theta\,\mathrm{KL}_t(y_{<t})\Big|_{\text{prefix 고정}} ,
$$

즉 **그 위치 조건부 KL의 gradient**입니다(같은 항등식을 토큰 하나짜리 분포에 적용한 것입니다). 조건부 분포는 어휘 크기 $|\mathcal{V}|$짜리 벡터이니, 학생과 선생의 **전체 logit**이 있으면 이 KL을 샘플 없이 정확히 계산해 그대로 backprop할 수 있습니다:

$$
\mathrm{KL}_t(y_{<t})=\sum_{v\in\mathcal{V}}\pi_\theta(v\mid y_{<t})\Big[\log\pi_\theta(v\mid y_{<t})-\log\pi_T(v\mid y_{<t})\Big].
$$

GKD·MiniLLM·Qwen3가 이 방식입니다. 분산이 없는 대신 선생의 full logit이 필요하고(같은 GPU에 선생을 띄워야 하고), 어휘가 $10^5$개면 위치당 $10^5$개의 곱셈이 붙습니다.

**estimator B — 샘플된 토큰 하나.** 선생에게서 **샘플된 토큰의 log-prob 하나**만 받습니다. 위 항등식 덕분에

$$
\hat g_t=\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})\cdot\big(\log\pi_\theta(y_t\mid y_{<t})-\log\pi_T(y_t\mid y_{<t})\big)
$$

는 $\nabla_\theta\mathrm{KL}_t$의 **unbiased estimator**입니다(기대값이 정확히 A와 같습니다). 선생은 forward pass 한 번으로 학생 응답 전체에 log-prob을 매기면 되고, 그 값은 API 뒤에 있는 모델에서도 받을 수 있습니다. Thinking Machines 레시피가 이쪽입니다.

### 3.5 reward-to-go를 자를 것인가 — discount 0의 의미

§3.4의 정확한 gradient에는 $s>t$ 항, 즉 "토큰 $t$를 이렇게 골랐더니 **뒤의 토큰들**이 선생과 얼마나 멀어졌나"가 들어 있습니다. 이것은 §3.2 분해에서 **prefix 분포 $y_{<s}\sim\pi_\theta$가 $\theta$에 의존**하는 데서 오는 항입니다. Thinking Machines는 이 항을 **버립니다** — RL 용어로 discount factor $\gamma=0$, advantage를 $A_t=-\rho_t$ 한 토큰짜리로 둡니다.

$$
\nabla_\theta\mathcal{L}\;\approx\;\mathbb{E}_{y\sim\pi_\theta}\Big[\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})\cdot\rho_t(y)\Big]
=\sum_{t}\mathbb{E}_{y_{<t}\sim\pi_\theta}\Big[\nabla_\theta\mathrm{KL}_t(y_{<t})\Big|_{\text{prefix 고정}}\Big].
$$

이것은 시퀀스 KL의 gradient로서는 **biased**입니다(뒤 토큰에 미치는 영향을 무시). 대신 분산이 크게 줄고 — reward-to-go는 길이 $L$짜리 합이라 분산이 $L$에 비례해 커집니다 — 각 위치에서 "지금 이 토큰의 분포를 선생 쪽으로 옮겨라"는 국소 신호만 남습니다. 무시한 항이 정말 작은가에 대한 정당화는 이렇습니다. 학습이 진행되어 $\pi_\theta\approx\pi_T$가 되면 모든 $\rho_s\to0$이므로 reward-to-go 항 전체가 0에 수렴하고, 수렴점은 두 estimator가 같습니다. 초기에 학생이 선생과 많이 다를 때는 뒤 토큰의 영향이 유의미하지만, 그때는 국소 신호만으로도 갈 방향이 뚜렷합니다. 요컨대 **"prefix가 학생 것"이라는 on-policy의 이점은 챙기고, "prefix가 $\theta$에 의존한다"는 미분의 골칫거리는 버린** 절충입니다. estimator A도 사실상 같은 근사를 씁니다 — 위치별 KL을 계산할 때 prefix는 상수로 두고 미분하니까요.

### 3.6 실제 loss 구현

샘플을 한 번 뽑아 여러 gradient step에 쓰면(mini-batch 재사용), 샘플링 시점의 정책 $\pi_{\text{old}}$와 현재 $\pi_\theta$가 달라집니다. 그래서 importance sampling 비율을 곱한 PPO/GRPO식 loss로 씁니다. 응답 $y\sim\pi_{\text{old}}$에 대해

$$
\mathcal{L}(\theta)=-\frac{1}{L}\sum_{t=1}^{L}
\frac{\pi_\theta(y_t\mid y_{<t})}{\pi_{\text{old}}(y_t\mid y_{<t})}\cdot A_t,\qquad
A_t=\log\pi_T(y_t\mid y_{<t})-\log\pi_{\text{old}}(y_t\mid y_{<t}).
$$

첫 step에서는 $\pi_\theta=\pi_{\text{old}}$라 비율이 1이고 gradient는 정확히 $\hat g_t$의 부호를 뒤집은 것($A_t=-\rho_t$이므로 loss 최소화 = KL 최소화)입니다. 필요하면 GRPO처럼 비율에 clipping을 걸고, 여러 응답을 뽑아 평균합니다. 의사코드로는:

```python
# prompts: batch of x;  student: pi_theta;  teacher: pi_T (frozen, forward only)
y, logp_old = student.sample(prompts)            # on-policy rollout, per-token log-probs
logp_T      = teacher.logprobs(prompts, y)        # ONE forward pass, same tokenizer required
A           = logp_T - logp_old                   # per-token advantage = -(reverse-KL sample)
for _ in range(k_steps):                          # a few steps on the same rollout
    logp   = student.logprobs(prompts, y)
    ratio  = (logp - logp_old).exp()
    loss   = -(ratio * A).mean()                  # (optionally clip ratio; optionally A -= baseline)
    loss.backward(); optimizer.step()
```

몇 가지를 짚어 둡니다.

- **baseline이 필요 없습니다.** RL에서 baseline은 보상의 절대 수준이 의미 없어서 빼 주는 것인데, $A_t$는 이미 자연스러운 영점이 있습니다 — 학생이 그 토큰에서 선생과 같으면 0, 선생이 더 좋아하면 양수, 학생만 좋아하면 음수. 기대값 $\mathbb{E}[A_t]=-\mathrm{KL}_t\le0$이라 평균적으로는 "확률을 낮춰라"가 많습니다. 원한다면 배치 평균을 빼도 되고(분산만 줄고 기대 gradient는 §3.4의 항등식으로 불변), Thinking Machines는 별도 정규화 없이 씁니다.
- **선생은 학습하지 않습니다.** $A_t$에 $\theta$가 들어 있지만 상수로 취급합니다(detach). 그것이 §3.4에서 둘째 항이 0이 되는 근거였습니다.
- **같은 tokenizer가 필요합니다.** $\log\pi_T(y_t\mid y_{<t})$를 잴 수 있으려면 학생이 만든 토큰열을 선생이 그대로 읽을 수 있어야 합니다. 어휘가 다르면 토큰 정렬 문제가 생겨 이 정확한 형태는 무너집니다.
- **선생 비용은 forward 한 번입니다.** SFT는 선생이 **생성**을 해야 하고(수천 토큰의 autoregressive decoding), OPD는 학생이 생성하고 선생은 **채점**만 합니다. prefill/decode 편의 언어로, 선생 쪽은 compute-bound prefill 한 번이 전부입니다.

## 4. 일반화 — GKD·MiniLLM·Thinking Machines의 자리

같은 틀 안에서 세 축을 돌릴 수 있습니다: **샘플을 누가 만드나**(선생/학생/혼합), **거리를 무엇으로 재나**(forward/reverse/사이), **gradient를 어떻게 추정하나**(A/B).

| 방법 | 샘플 분포 | divergence | estimator | 비고 |
|---|---|---|---|---|
| SFT / sequence-level KD (Kim & Rush 2016) | $\pi_T$ | forward KL | 정확(cross-entropy) | 분포 불일치 |
| RL (RLVR·GRPO) | $\pi_\theta$ | — (보상 $R$) | score-function + baseline | 시퀀스당 1 신호, verifier 필요 |
| MiniLLM (Gu et al. 2023) | $\pi_\theta$ (선생 혼합) | reverse KL | score-function, 길이 정규화 | 단일 step 분해 + 장기 항 |
| GKD (Agarwal et al. 2023) | $\lambda\pi_\theta+(1-\lambda)\pi_T$ | $\mathrm{JSD}(\beta)$ | A (full logit) | $\lambda,\beta$로 위 방법들을 보간 |
| Qwen3 (2025) | $\pi_\theta$ | reverse KL(logit) | A | strong-to-weak 단계에 사용 |
| Thinking Machines OPD (2025) | $\pi_\theta$ | reverse KL | B, $\gamma=0$, IS 비율 | API 선생 가능, RL 인프라 재사용 |

GKD의 **generalized JSD**는 두 방향을 잇는 손잡이입니다. $M_\beta=\beta\pi_\theta+(1-\beta)\pi_T$라 두고

$$
\mathrm{JSD}_\beta(\pi_\theta\|\pi_T)=\beta\,\mathrm{KL}(\pi_\theta\|M_\beta)+(1-\beta)\,\mathrm{KL}(\pi_T\|M_\beta),
$$

$\beta\to1$이면 reverse KL, $\beta\to0$이면 forward KL에 가까워집니다(정확히는 상수배 극한). GKD의 실험 결론은 "on-policy 샘플($\lambda$ 큼) + mode-seeking 쪽($\beta$ 큼)이 요약·번역·산술 모두에서 낫다"였고, 그것이 이후 모든 OPD 레시피의 기본값이 됐습니다. MiniLLM은 reverse KL을 estimator B로 풀되 reward-to-go를 **단일 step 항 + 장기 항**으로 나눠 앞의 것은 정확히(A처럼), 뒤의 것만 샘플로 추정했습니다 — §3.5에서 Thinking Machines가 버린 항을 MiniLLM은 남긴 셈입니다. 어느 쪽이 맞는지는 아직 실험적 문제입니다.

## 5. 왜 싸고 왜 잘 되나 — 비용과 정보량

응답 하나를 학습에 쓰는 비용을 세어 봅시다. $C_{\text{gen}}$은 길이 $L$ 응답을 생성하는 비용(decode, memory-bound), $C_{\text{fwd}}$는 같은 길이를 한 번 forward하는 비용(prefill), $C_{\text{bwd}}\approx2C_{\text{fwd}}$입니다. 첨자 $S,T$는 학생·선생.

| | 생성 | 채점/보상 | 학생 학습 | 응답당 정보량 |
|---|---|---|---|---|
| SFT | $C^{T}_{\text{gen}}$ (선생, 비쌈) | — | $C^{S}_{\text{fwd}}+C^{S}_{\text{bwd}}$ | $L$개 토큰 목표(선생 문맥) |
| RL | $C^{S}_{\text{gen}}$ | verifier/RM | $C^{S}_{\text{fwd}}+C^{S}_{\text{bwd}}$ | 스칼라 1개 |
| OPD | $C^{S}_{\text{gen}}$ | $C^{T}_{\text{fwd}}$ | $C^{S}_{\text{fwd}}+C^{S}_{\text{bwd}}$ | $L$개 스칼라(학생 문맥) |

RL과 OPD는 rollout 비용이 같고, OPD는 선생 forward가 추가됩니다. 그런데 응답당 정보량이 스칼라 1개에서 $L$개로 늘어나니 **필요한 rollout 수가 크게 줄어듭니다.** Thinking Machines의 보고(Qwen3-8B-Base 학생, Qwen3-32B 선생, AIME'24)는 대략 이렇습니다 — SFT 40만 프롬프트로 60% 근처였던 것을 OPD 150 step에 70%로 올렸고, 같은 점수까지 가는 GPU 시간이 SFT 외삽 대비 한 자릿수에서 수십 배 절약, RL 대비도 비슷한 규모의 절약(정확한 배수는 원문 재확인을 권합니다). 핵심 이유는 세 가지로 요약됩니다.

1. **dense signal**: §3.2의 분해가 토큰마다 gradient 방향을 줍니다. RL의 credit assignment 문제가 사라집니다.
2. **on-policy**: 학생이 실제로 저지르는 실수의 문맥에서 배웁니다. §2.1의 $L^2\to L$.
3. **샘플 재사용**: RL은 같은 프롬프트를 여러 epoch 돌리면 금방 과적합하지만, OPD는 선생이 매번 새 문맥에서 채점하므로 **프롬프트가 한 개뿐이어도** 학습이 진행됩니다(Thinking Machines의 극단 실험).

여기에 **선생이 꼭 더 큰 모델일 필요가 없다**는 점이 응용을 넓힙니다. Thinking Machines의 두 번째 실험은 continual learning입니다 — 사내 문서로 mid-training한 모델은 지식은 늘지만 instruction following이 무너지는데, **mid-training 전의 자기 자신**을 선생으로 OPD를 돌리면 지식은 유지한 채 행동이 복구됩니다. reverse KL이 "선생이 하지 않을 행동"을 지우는 zero-forcing이기 때문에, 지식 토큰(선생도 모르니 확률 차이가 작음)은 건드리지 않고 행동 붕괴(선생이 강하게 반대)만 고치는 셈입니다.

## 6. 제약과 함정

- **선생의 off-distribution 채점.** $\pi_T(y_t\mid y_{<t})$는 **학생의** prefix 위에서 평가됩니다. 학생이 이상한 prefix를 만들면 선생도 본 적 없는 문맥에서 채점하는 것이라 그 값이 얼마나 믿을 만한지는 보장이 없습니다. 선생이 학생보다 훨씬 강할 때 잘 되는 이유이기도, 학생이 너무 형편없을 때(초기 base 모델) SFT로 워밍업을 먼저 하는 이유이기도 합니다.
- **다양성 붕괴.** mode-seeking은 곧 "선생의 답 중 하나"로 수렴한다는 뜻입니다. pass@$k$처럼 다양성이 중요한 지표나, OPD 뒤에 RL을 이어 붙일 때의 탐색 여력은 줄 수 있습니다. GKD가 $\beta$를 남겨 둔 이유입니다.
- **선생 결함의 복제.** 보상이 $\log\pi_T$이므로 선생이 좋아하는 것은 무엇이든 배웁니다 — 선생이 반복적 표현에 높은 확률을 주면 학생도 그렇게 됩니다. RL의 reward hacking과 달리 "보상 모델을 속이는" 형태는 아니지만, 선생의 편향은 그대로 전이됩니다.
- **길이 편향.** $\sum_t\rho_t$는 길이에 따라 커지므로 길이 정규화($1/L$)를 하지 않으면 짧은 응답이 유리해집니다. MiniLLM이 정규화를 명시한 이유입니다.
- **$\gamma=0$의 맹점.** 지금 토큰의 선택이 뒤에서 큰 문제를 일으키는 경우(예: 잘못된 형식을 열어 놓고 닫지 못함)를 국소 신호는 보지 못합니다. 실전에서는 SFT→OPD→RL처럼 sparse-but-global 신호와 섞어 씁니다.

## 7. 결론

OPD의 loss는 **reverse KL** $\mathrm{KL}(\pi_\theta\|\pi_T)$ 하나입니다. 그것이 §3.2의 chain rule 분해로 "학생 prefix 위 토큰별 KL의 합"이 되고, §3.4의 score-function 항등식으로 "보상이 $\log\pi_T-\log\pi_\theta$인 policy gradient"가 됩니다. 그 순간 RL 인프라(rollout·importance sampling·clipping)를 그대로 재사용하면서, reward model 없이 토큰마다 dense한 신호를 얻습니다. SFT의 분포 불일치와 RL의 sparse reward를 동시에 없앤 대가는 "같은 tokenizer의 선생이 있어야 한다"와 "선생이 학생의 문맥에서 채점한다"는 두 가지 조건뿐입니다. 학생이 직접 쓰고, 선생은 토큰마다 채점한다 — 그것이 전부이고, 그래서 통합니다.

*(참고: Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"(GKD), arXiv:2306.13649 · Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models", arXiv:2306.08543 · Thinking Machines Lab, "On-Policy Distillation", 2025년 10월 블로그(Kevin Lu 외) · Qwen3 Technical Report, arXiv:2505.09388 · Kim & Rush, "Sequence-Level Knowledge Distillation", 2016 · Ross & Bagnell, "Efficient Reductions for Imitation Learning", AISTATS 2010. 인용한 실험 수치는 각 원문의 대략값이며 정확한 값은 원문을 확인하세요. 그림은 직접 그린 것입니다.)*

#OnPolicyDistillation #distillation #reverseKL #RL #정렬·추론
