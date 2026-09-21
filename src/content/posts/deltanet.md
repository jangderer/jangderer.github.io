---
title: "기억을 덮어쓰는 법 — delta rule"
subtitle: "DeltaNet — 선형 어텐션의 덧셈 기억을 '지우고 쓰기'로 바꾼 Schlag의 delta rule과, 그것을 Householder 곱·WY 표현으로 청크 병렬화한 Yang의 알고리즘"
description: "선형 어텐션의 상태 S에 vkᵀ를 더하기만 하면 키가 d개를 넘는 순간 간섭이 쌓인다. Delta rule은 현재 키로 옛 값을 먼저 읽어 빼고 새 값을 쓴다 — 온라인 회귀 손실 ½‖Sk−v‖²의 SGD 한 걸음이고, 전이 행렬 I−βkkᵀ는 k 방향 고유값 1−β를 갖는 일반화 Householder다. 이 전이가 대각이 아니라서 청크 병렬화가 막혔던 것을 WY 표현과 UT 변환(삼각 행렬 풀기)으로 뚫은 유도, 1.3B/100B 결과에서 delta rule이 실제로 이기는 과제와 지는 과제, β를 2σ로 바꾸면 parity가 풀리는 이유까지."
pubDate: 2026-09-21
order: 25
track: "아키텍처"
tags: ["DeltaNet","delta rule","linear attention","fast weights","Householder","아키텍처"]
hero: "/images/deltanet.png"
---

Linear attention 편의 마지막 절에서 한 줄로 지나간 식이 있습니다.

$$
S_t = S_{t-1}(I - \beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}
$$

그때는 "기존 기억에서 관련 성분을 지우고 새로 덮어쓴다"고만 적었습니다. 이 글은 그 한 줄을 처음부터 다시 씁니다. 왜 덧셈만 하는 기억이 실패하는지, delta rule이 정확히 무엇을 지우는지, 그 전이 행렬이 대각이 아니라서 생기는 병렬화 문제를 어떻게 풀었는지, 그리고 이 규칙이 실제로 언어 모델에서 무엇을 이기고 무엇을 지는지.

두 논문이 축입니다. Schlag·Irie·Schmidhuber(2021)는 선형 어텐션이 1990년대의 fast weight programmer와 같은 것임을 보이고 delta rule을 처방했고, Yang·Wang·Zhang·Shen·Kim(2024)은 그것을 시퀀스 방향으로 병렬화해 실제로 학습 가능하게 만들었습니다. 뒤에 나오는 Gated DeltaNet과 Qwen3-Next·Kimi Linear의 선형 층은 전부 이 두 논문 위에 서 있습니다.

## 1. 덧셈 기억의 용량 — 왜 실패하는가

Linear attention 편의 재귀 형태를 다시 씁니다. 상태 $S_t \in \mathbb{R}^{d_v \times d_k}$는 값을 행, 키를 열로 갖는 행렬이고, 매 토큰 외적을 더하며, 읽기는 query와의 곱입니다.

$$
S_t = S_{t-1} + v_t k_t^{\top}, \qquad o_t = S_t q_t
$$

Schlag 등은 이것이 Schmidhuber(1991)의 fast weight programmer와 같은 것임을 지적했습니다. $S$는 느린 가중치 $W_K, W_V$가 매 토큰 "프로그래밍"하는 빠른 가중치이고, 프로그래밍 명령은 외적 덧셈(Hebb 규칙)입니다.

이 기억의 용량은 키의 차원이 정합니다. 키 $k_1, \ldots, k_n$으로 값들을 저장한 뒤 $k_1$로 읽으면

$$
S k_1 = \sum_{i=1}^{n} v_i (k_i^{\top} k_1) = v_1 \|k_1\|^2 + \sum_{i \ne 1} v_i (k_i^{\top} k_1)
$$

두 번째 항이 간섭입니다. 간섭이 0이려면 키들이 서로 직교해야 하는데, $d_k$차원에는 직교 벡터가 $d_k$개뿐입니다. Schlag 등의 표현으로 "$d_{\mathrm{dot}}$개보다 많은 연관을 저장하면 검색 오류가 난다. 선형 트랜스포머는 시퀀스가 $d_{\mathrm{dot}}$보다 길면 이런 과용량 상태에 있을 수 있다." 위 그림 왼쪽이 이것을 $d = 64$에서 재현한 것입니다. 무작위 단위 키로 쌍을 계속 쓰면 첫 쌍의 검색 오류가 멈추지 않고 자랍니다.

문제는 덧셈에 있습니다. 같은 키로 두 번 쓰면 두 값이 **합쳐지지** 대체되지 않습니다. 기억을 갱신하려면 먼저 그 자리에 무엇이 있는지 읽고 지워야 합니다.

![DeltaNet: interference simulation and transition eigenvalue](/images/deltanet.png)

> 왼쪽: $d = 64$에서 무작위 단위 키로 쌍을 계속 쓸 때 첫 쌍의 검색 오류 $\|Sk_1 - v_1\|$ (20회 평균, 직접 시뮬레이션). 덧셈 기억은 오류가 멈추지 않고 자라고, delta rule은 각 쓰기가 자기 키 방향의 옛 값만 지우므로 유계에 머문다. 오른쪽: 전이 $I - \beta kk^{\top}$이 $k$ 방향의 옛 기억에 곱하는 계수 $1 - \beta$. $\beta = \sigma(\cdot) \in (0, 1)$이면 양수, $\beta = 1$이면 정확한 덮어쓰기, $\beta = 2\sigma(\cdot)$로 확장하면 음의 고유값이 허용된다(6절). 직접 그린 그림이다.

## 2. Delta rule — 읽고, 빼고, 쓴다

Delta rule은 정확히 그 순서로 동작합니다. 현재 키로 옛 값을 읽고, 새 값과 섞고, 옛 값을 지운 자리에 씁니다.

$$
v^{\mathrm{old}}_t = S_{t-1} k_t, \qquad v^{\mathrm{new}}_t = \beta_t v_t + (1 - \beta_t)\, v^{\mathrm{old}}_t
$$
$$
S_t = S_{t-1} - v^{\mathrm{old}}_t k_t^{\top} + v^{\mathrm{new}}_t k_t^{\top}
$$

$\beta_t = \sigma(w_\beta^{\top} x_t) \in (0, 1)$은 토큰이 정하는 **쓰기 강도**입니다. $\beta_t = 1$이면 옛 값이 완전히 지워지고 $v_t$가 들어가며, $\beta_t = 0$이면 기억이 그대로입니다. 두 줄을 합치면 $v^{\mathrm{new}} - v^{\mathrm{old}} = \beta_t(v_t - v^{\mathrm{old}}_t)$이므로

$$
S_t = S_{t-1} + \beta_t\,(v_t - S_{t-1} k_t)\, k_t^{\top}
$$

이것이 delta rule입니다. 이름은 "예측 오차 $\delta = v_t - S_{t-1}k_t$에 비례해 갱신한다"는 Widrow–Hoff의 규칙에서 왔습니다. 항을 다시 묶으면 글 첫머리의 형태가 됩니다.

$$
S_t = S_{t-1}\big(I - \beta_t k_t k_t^{\top}\big) + \beta_t v_t k_t^{\top}
$$

### 2.1 온라인 경사 하강으로서

이 갱신은 손실 하나의 SGD 한 걸음과 같습니다. 손실을 "이 키로 읽으면 이 값이 나와야 한다"는 회귀로 두면

$$
\mathcal{L}_t(S) = \tfrac{1}{2}\|S k_t - v_t\|^2
$$

기울기를 구하기 위해 $r = Sk_t - v_t$로 두면 $d\mathcal{L} = r^{\top} dr = r^{\top}(dS)k_t = \mathrm{tr}(k_t r^{\top} dS)$이고, $d\mathcal{L} = \mathrm{tr}(G^{\top} dS)$가 기울기 $G$를 정의하므로 $G = r k_t^{\top}$입니다. 따라서

$$
S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t(S_{t-1}) = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^{\top}
$$

로 delta rule과 정확히 같습니다. 같은 관점에서 선형 어텐션은 손실 $-\langle S k_t, v_t \rangle$의 SGD입니다. 그 기울기는 $-v_t k_t^{\top}$로 **오차와 무관하게 상수**입니다. 이미 맞는 값이 저장되어 있어도 계속 더하고, 그것이 간섭이 쌓이는 이유입니다. 제곱 손실은 오차가 0이면 갱신이 0입니다.

### 2.2 전이 행렬의 스펙트럼

$I - \beta k k^{\top}$은 항등 더하기 rank-1 행렬, 즉 **일반화된 Householder 행렬**입니다. $k$에 곱하면 $(I - \beta k k^{\top})k = (1 - \beta\|k\|^2)k$이고, $k$에 직교하는 $u$에 곱하면 $u$ 그대로입니다. 고유값은 $k$ 방향으로 $1 - \beta\|k\|^2$ 하나, 나머지 $d - 1$개 방향으로 1입니다.

그래서 키의 크기가 중요합니다. Schlag 등은 특성 사상 $\phi$의 출력을 합으로 나눠(양수이므로 L1 정규화) $\|k\|_2 \le 1$을 보장했고, Yang 등은 L2 정규화로 바꿨습니다. $\|k\|_2 = 1$이면 고유값은 정확히 $1 - \beta$이고, $\beta = 1$일 때 $I - kk^{\top}$은 **사영 행렬**이 됩니다. $k$ 방향의 기억만 완전히 지우고 나머지 $d - 1$차원은 손대지 않는 것입니다. 이 해석이 위 그림 오른쪽이고, ablation에서 L1을 L2로 바꾸는 것만으로 340M 모델의 LAMBADA perplexity가 55.96에서 37.62로 내려갑니다.

행렬은 대칭이므로 스펙트럼 노름은 $\max(1, |1 - \beta|)$이고, $0 \le \beta \le 2$일 때 비확장입니다. Yang 등은 $\beta = \sigma(\cdot) \in (0, 1)$로 두어 고유값을 $(0, 1)$에 가둡니다. 이 선택이 나중에 문제가 되는데, 6절에서 봅니다.

### 2.3 게이트와 무엇이 다른가

Schlag 등의 부록에 있는 작은 예가 이 규칙의 요점을 가장 잘 보여 줍니다. 정규직교 키 $k_1, k_2$로 $v_1, v_2$를 저장한 뒤 $k_3 = k_2$로 $v_3$를 씁니다. 게이트 규칙 $S' = (1 - \beta)S + \beta v_3 k_3^{\top}$은 $S'k_1 = (1 - \beta)v_1$로 **무관한 연관까지 감쇠**시킵니다. Delta rule은 $S'k_1 = v_1$을 그대로 두고 $S'k_3 = (1 - \beta)v_2 + \beta v_3$로 해당 연관만 갱신합니다. 게이트는 기억 전체를 잊고, delta rule은 한 방향만 덮어씁니다. 이 두 가지가 서로 다른 축이라는 것이 Gated DeltaNet 편의 출발점입니다.

## 3. 병렬화 — 대각이 아닌 전이의 문제

여기까지가 2021년입니다. Schlag 등의 구현은 토큰을 하나씩 순차로 처리하는 CUDA 커널이었고, Yang 등의 표현으로 "엄격히 순차적이라 하드웨어 비효율적"이었습니다.

Linear attention 편의 청크 알고리즘을 떠올려 봅니다. 상태 갱신이 $S_t = S_{t-1} + v_t k_t^{\top}$처럼 덧셈이면 청크 안의 기여를 행렬 곱 하나로 합칠 수 있습니다. GLA나 Mamba-2처럼 전이가 **대각 행렬**(원소별 감쇠)이면 누적 곱이 원소별 곱이라 역시 쉽습니다. Delta rule의 전이 $I - \beta_t k_t k_t^{\top}$은 대각이 아닙니다. 토큰 $i$의 기여가 이후 모든 전이의 곱을 통과해야 합니다.

$$
S_t = \sum_{i=1}^{t} \beta_i (v_i k_i^{\top}) \prod_{j=i+1}^{t}\big(I - \beta_j k_j k_j^{\top}\big)
$$

이 곱을 그냥 계산하면 토큰마다 $d \times d$ 행렬 곱이라 $O(Ld^2)$이고 순차적입니다. Yang 등의 기여는 이 곱이 특별한 구조를 가진다는 것을 이용한 것입니다.

### 3.1 덧셈 형태로의 재매개화

먼저 관찰 하나. Delta rule은 "지우고 쓰기"지만, 지우는 항도 결국 $k_t^{\top}$에 붙은 외적이므로 전체를 외적의 합으로 다시 쓸 수 있습니다.

$$
S_t = \sum_{i=1}^{t} u_i k_i^{\top}, \qquad u_t = \beta_t\Big(v_t - \sum_{i=1}^{t-1} u_i (k_i^{\top} k_t)\Big)
$$

귀납으로 확인하면, $S_{t-1} = \sum_{i<t} u_i k_i^{\top}$일 때 $S_{t-1}k_t = \sum_{i<t} u_i(k_i^{\top}k_t)$이므로 $\beta_t(v_t - S_{t-1}k_t) = u_t$이고, $S_t = S_{t-1} + u_t k_t^{\top}$입니다. $u_t$는 "옛 값을 뺀 뒤의 실제 쓰기량", 논문 용어로 **의사 값**(pseudo-value)입니다. 의사 값만 구해 놓으면 나머지는 보통의 선형 어텐션입니다. $O = (QK^{\top} \odot M)\,U$. 문제는 $u_t$ 자체가 이전 $u_i$ 전부에 의존해 $O(L^2 d)$이고 순차적이라는 것입니다. 그래서 청크로 자릅니다.

### 3.2 WY 표현 — Householder 곱은 rank-$r$이다

청크 크기 $C$(기본 64)로 자르고, 청크 안에서 $r$개의 전이를 곱한 것을 $P^r = \prod_{i=1}^{r}(I - \beta_i k_i k_i^{\top})$이라 합니다. 수치선형대수의 고전 결과(Bischof–Van Loan 1985, WY 표현)는 Householder 행렬 $r$개의 곱이 **항등 더하기 rank-$r$ 행렬**이라는 것입니다.

$$
P^r = I - \sum_{i=1}^{r} w_i k_i^{\top}, \qquad w_r = \beta_r\Big(k_r - \sum_{i=1}^{r-1} w_i (k_i^{\top} k_r)\Big)
$$

귀납으로 유도하면 $P^r = P^{r-1}(I - \beta_r k_r k_r^{\top}) = (I - \sum_{i<r} w_i k_i^{\top})(I - \beta_r k_r k_r^{\top})$이고, 전개하면 $I - \sum_{i<r} w_i k_i^{\top} - [\beta_r k_r - \beta_r \sum_{i<r} w_i(k_i^{\top}k_r)]\,k_r^{\top}$로 대괄호가 $w_r$입니다. 같은 구조로 청크 안의 누적 쓰기 $H^r = \sum_{i \le r} \beta_i (v_i k_i^{\top}) P^{i+1..r}$도 $H^r = \sum_{i=1}^{r} u_i k_i^{\top}$이며 $u_r$은 위 $w_r$ 식에서 $k_r$을 $v_r$로 바꾼 것입니다.

즉 청크 하나의 효과는 $d \times d$ 행렬이 아니라 $C$개의 벡터 쌍 $(w_i, k_i)$와 $(u_i, k_i)$로 표현됩니다. 저장은 $O(Cd)$, 적용은 행렬 곱입니다.

### 3.3 UT 변환 — 재귀를 삼각 행렬 풀기로

$w_r$의 식은 여전히 재귀입니다. $r$번째를 구하려면 앞의 $r - 1$개가 필요합니다. 이를 행렬로 쓰면 재귀가 **삼각 선형계**가 됩니다. $W \in \mathbb{R}^{C \times d}$를 $w_i$를 행으로 쌓은 것, $B = \mathrm{diag}(\beta)$, $L = \mathrm{tril}(BKK^{\top}, -1)$(엄격 하삼각)로 두면 $w_r$의 식은 행 단위로 $W[r,:] = \beta_r K[r,:] - \sum_{i<r} \beta_r (k_i^{\top}k_r) W[i,:]$, 즉

$$
(I + L)\,W = BK \quad\Longrightarrow\quad W = TK,\;\; U = TV, \qquad T = \big(I + \mathrm{tril}(\mathrm{diag}(\beta)KK^{\top}, -1)\big)^{-1}\mathrm{diag}(\beta)
$$

$T$는 $C \times C$ 하삼각 행렬의 역행렬이고, 하삼각의 역행렬은 전진 대입으로 $O(C^2)$ 행마다 풀 수 있으며 텐서 코어에 올릴 수 있습니다. 이것이 UT 변환(Joffrain et al. 2006)입니다. 초기 arXiv 판의 이 식에는 오류가 있었고 v6에서 수정되었습니다.

### 3.4 청크 재귀

이제 청크 $[t]$의 입력 상태를 $S_{[t]}$라 하면, 청크 끝의 상태와 청크 안의 출력은

$$
S_{[t+1]} = S_{[t]} + \big(U_{[t]} - W_{[t]} S_{[t]}^{\top}\big)^{\top} K_{[t]}
$$
$$
O_{[t]} = Q_{[t]} S_{[t]}^{\top} + \big(Q_{[t]}K_{[t]}^{\top} \odot M\big)\big(U_{[t]} - W_{[t]} S_{[t]}^{\top}\big)
$$

Linear attention 편의 청크 식 $S_{[t+1]} = S_{[t]} + V_{[t]}^{\top}K_{[t]}$, $O_{[t]} = Q_{[t]}S_{[t]}^{\top} + (Q_{[t]}K_{[t]}^{\top} \odot M)V_{[t]}$와 나란히 놓으면 차이가 정확히 하나입니다. **값 $V$가 보정된 의사 값 $U - WS^{\top}$로 바뀌었습니다.** $U$는 청크 안에서 서로 지운 결과이고, $WS^{\top}$은 이 청크의 Householder 곱이 들어오는 상태에 가한 지우기입니다. DeltaNet의 청크 알고리즘은 "값을 먼저 보정한 선형 어텐션"입니다.

복잡도는 청크당 $KK^{\top}$ $O(C^2 d)$, 삼각 풀기 $O(C^2)$~$O(C^3)$, $TK$·$TV$ $O(C^2 d)$, 상태 곱 $O(Cd^2)$로 전체 $O(L(Cd + d^2 + C^2))$입니다. $C \le d$이면 선형 어텐션과 같은 차수이고 상수가 큽니다. 논문도 "학습 속도는 여전히 GLA에 뒤진다"고 적습니다. 공개 커널(flash-linear-attention)은 $KK^{\top}$과 삼각 역행렬을 fp32로 계산한 뒤 $T$를 bf16으로 내려 $TK$, $TV$를 곱하며, 청크 크기는 16·32·64만 허용하고 fp32 입력은 받지 않습니다.

## 4. 층 구성

Yang 등의 DeltaNet 층은 LLaMA식 블록에서 attention만 바꾼 것입니다. 사영 $W_Q, W_K, W_V$ 뒤에 커널 4의 짧은 causal conv를 두고, query와 key에는 SiLU 뒤 L2 정규화를 적용합니다.

$$
k_t = \frac{\mathrm{SiLU}(W_K x_t)}{\|\mathrm{SiLU}(W_K x_t)\|_2}, \qquad q_t = \frac{\mathrm{SiLU}(W_Q x_t)}{\|\mathrm{SiLU}(W_Q x_t)\|_2}, \qquad \beta_t = \sigma(w_\beta^{\top} x_t)
$$

$\beta$는 헤드당 스칼라 하나이고, 헤드 차원은 128, 출력은 헤드별 RMSNorm 뒤 사영입니다. GLA와 달리 출력 게이트는 기본으로 두지 않습니다. 짧은 conv는 ablation에서 결정적입니다. 340M에서 conv를 빼면 LAMBADA perplexity가 37.37에서 50.87로 오르고, 공개 코드 주석은 "ShortConvolution is crucial to the performance. Do not turn it off"입니다. 파라미터 예산은 층당 $4d^2$로 attention과 같습니다.

## 5. 결과 — 어디서 이기고 어디서 지는가

1.3B 모델을 SlimPajama 100B 토큰으로 학습한 결과(Table 1)입니다. 상식 과제 평균과 회상 과제 세 개를 같이 봐야 합니다.

| 1.3B / 100B | Wiki ppl | LMB ppl | 상식 평균 | SWDE | SQuAD | FDA | 상태 배율 |
|---|---|---|---|---|---|---|---|
| Transformer++ | 16.85 | 13.44 | 50.9 | 66.6 | 31.5 | 27.4 | — |
| Mamba | 17.06 | 13.89 | 50.0 | 41.4 | 35.2 | 6.2 | 64× |
| GLA | 17.22 | 14.47 | 51.0 | 50.6 | 42.6 | 19.9 | 256× |
| DeltaNet | 16.87 | 12.21 | 51.6 | 49.5 | 37.4 | 17.2 | 128× |
| + Sliding Attn | 16.56 | 11.74 | 52.1 | 53.3 | 43.3 | 22.3 | — |
| + Global Attn (2층) | 16.55 | 12.40 | 51.8 | 71.0 | 43.0 | 29.8 | — |

세 가지가 읽힙니다.

첫째, **perplexity와 상식 과제에서 DeltaNet은 Transformer++를 앞섭니다.** LAMBADA perplexity 12.21은 표에서 순수 모델 중 가장 낮고, 평균 51.6도 Transformer++의 50.9보다 높습니다. 선형 시간 모델이 같은 예산에서 트랜스포머를 이기는 것은 이 시점에 드문 결과였습니다.

둘째, **회상 과제에서는 여전히 집니다.** 문서에서 정보를 추출하는 SWDE와 FDA에서 Transformer++의 66.6, 27.4에 DeltaNet은 49.5, 17.2입니다. 고정 크기 상태의 한계입니다. 그런데 340M에서는 DeltaNet이 GLA를 회상 과제에서 앞섰는데(FDA 12.8 대 7.3), 1.3B에서는 뒤집힙니다(17.2 대 19.9). 논문의 설명은 상태 크기입니다. 커널이 헤드 차원을 128로 제한해 DeltaNet의 상태 배율이 128×에 머무는 반면 GLA는 256×입니다. delta rule의 이점이 상태 크기의 열세에 묻힌 것입니다.

셋째, **하이브리드가 답입니다.** 전체 층의 딱 두 개(2번째와 $N/2 + 1$번째)를 전역 attention으로 바꾸면 SWDE 71.0, FDA 29.8로 Transformer++를 넘습니다. Hybrid 편에서 본 "회상은 attention 몇 층이면 충분하다"의 또 다른 증거입니다.

### 5.1 합성 과제 — delta rule이 정확히 잘하는 것

MAD 벤치마크의 여섯 과제에서 DeltaNet은 in-context recall, noisy recall, selective copy에서 **100**을 찍고, fuzzy recall에서 35.7로 다른 모든 모델(Transformer 29.8, Mamba 6.7, GLA 6.9)을 앞섭니다. Fuzzy recall은 키가 정확히 같지 않고 비슷한 경우인데, 제곱 손실의 기울기가 오차에 비례하므로 비슷한 키의 연관을 부드럽게 갱신하는 것과 맞습니다. 반면 memorize에서는 52.8로 Mamba(89.5)와 Transformer(85.2)에 크게 뒤집니다. 논문은 "이유는 불분명"이라고만 적습니다. 지우고 쓰는 규칙이 단순 암기에서는 오히려 방해가 되는 것으로 보이지만 분석은 없습니다.

### 5.2 3B

논문의 가장 큰 모델은 DeltaNet-3B, 1T 토큰입니다. 같은 설정의 트랜스포머 PowerLM-3B에 평균 59.8 대 62.3으로 뒤지고 MMLU에서 40.7 대 45.0입니다. Mamba-2.7B(53.3)와 RWKV-6-3B(54.9)보다는 앞서지만 학습 토큰이 달라 직접 비교는 아닙니다. 순수 DeltaNet이 3B에서 트랜스포머를 넘지는 못한다는 것이 정직한 요약입니다.

## 6. β의 범위 — parity가 풀리는 한 줄

Yang 등이 $\beta \in (0, 1)$로 둔 것은 안정성을 위한 선택이었지만, Grazzi·Siems 등(2024)은 이것이 표현력을 제한한다는 것을 보였습니다. 결과는 단순합니다. **전이 행렬의 고유값이 전부 음이 아니면 parity를 풀 수 없다.** Parity는 비트열의 1의 개수가 홀수인지 판단하는 문제이고, 상태가 두 개인 유한 자동자입니다. 그 전이는 "뒤집기"인데, 뒤집기를 선형 재귀로 구현하려면 어느 방향의 부호를 바꿔야 하고, 그것은 음의 고유값입니다. Mamba, GLA, mLSTM은 감쇠가 $(0, 1)$이므로 구조적으로 불가능하고, DeltaNet도 $\beta \in (0, 1)$이면 고유값 $1 - \beta \in (0, 1)$로 마찬가지입니다.

고치는 방법은 한 줄입니다. $\beta = 2\sigma(\cdot) \in (0, 2)$. 고유값이 $1 - \beta \in (-1, 1)$이 되고, $\beta = 2$이면 $I - 2kk^{\top}$은 정확한 Householder **반사**입니다. 노름은 여전히 1 이하라 안정성은 그대로입니다. 위 그림 오른쪽의 초록 영역이 이것입니다. 합성 실험에서 DeltaNet$[0,1]$의 parity 정확도 0.017이 DeltaNet$[-1,1]$에서 1.000이 되고, 괄호 없는 모듈러 산술은 0.314에서 0.971이 됩니다. Mamba에 같은 처방을 하면 parity는 풀리지만(1.000) 언어 모델링이 "일관되게 나빠지는" 반면, DeltaNet은 1.3B에서 perplexity 18.54 대 18.57로 거의 같습니다. Yang 등의 코드에는 이것이 `allow_neg_eigval` 옵션 하나로 들어가 있습니다.

Siems 등(2025)의 DeltaProduct는 토큰당 Householder를 $n_h$개 곱해 rank-$n_h$ 전이를 만듭니다. $S_5$(다섯 원소의 치환군) 단어 문제를 풀려면 $n_h = 4$가 필요하고, $n_h = 1$이면 10층으로도 안 됩니다. 이 문제는 트랜스포머와 대각 선형 RNN이 속하는 회로 복잡도 클래스 TC$^0$ 밖(NC$^1$)에 있다는 것이 그들의 위치 설정입니다.

## 7. 계보

Schmidhuber(1991)의 fast weight programmer가 외적 쓰기를, Katharopoulos 등(2020)의 선형 어텐션이 같은 것을 정규화와 함께 재발견했고, Schlag 등(2021)이 delta rule과 $\beta$를 붙여 DeltaNet이라 불렀습니다. Yang 등(2024)이 WY·UT로 청크 병렬화하고 SiLU+L2·짧은 conv·하이브리드를 더했으며, Grazzi 등이 $\beta \to 2\beta$, Siems 등이 $n_h$개 Householder로 확장했습니다. Gated DeltaNet은 여기에 Mamba-2식 스칼라 감쇠 $\alpha_t$를 곱해 $S_t = S_{t-1}\alpha_t(I - \beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}$로 만든 것이고, 다음 편의 주제입니다. RWKV-7은 감쇠를 벡터로, 지우는 키와 쓰는 키를 분리해 $S_t = S_{t-1}(\mathrm{diag}(w_t) - \hat{\kappa}_t^{\top}(a_t \odot \hat{\kappa}_t)) + v_t^{\top}k_t$로 일반화합니다.

순수 DeltaNet을 그대로 쓰는 제품 모델은 확인하지 못했습니다. 제품에 들어간 것은 전부 게이트가 붙은 변종입니다.

## 8. 논문이 말하지 않는 것

- **규모.** 주 비교는 1.3B/100B이고, 3B/1T 하나가 트랜스포머에 뒤집니다. 스케일링 법칙은 없습니다.
- **길이 외삽.** 논문 스스로 "DeltaNet의 길이 일반화는 제한적"이라며 "명시적 감쇠가 없어서"라고 추측합니다. Siems 등은 학습 길이를 넘으면 상태의 유효 rank가 계속 자라는 것을 원인으로 지목합니다. 2K를 넘는 문맥으로 학습한 결과는 없습니다.
- **속도 수치.** 청크 커널이 순차 커널보다 "최대 약 30배" 빠르다는 것과 학습 처리량이 "GLA에 가깝고 Mamba보다 훨씬 빠르다"는 것은 그림에만 있고 본문에 숫자가 없습니다.
- **삼각 풀기의 정밀도.** $T$를 fp32로 구한 뒤 bf16으로 내려 곱하는데, 청크가 크거나 $KK^{\top}$이 나쁘게 조건화될 때의 오차는 어디서도 분석되지 않습니다.
- **청크 크기 민감도.** 기본 64이고 "보통 64 또는 128"이라고만 합니다.
- **Memorize 과제의 부진.** 이유가 없습니다.
- **기준선.** Transformer++·RetNet·Mamba·GLA(conv 없음)의 수치는 GLA 논문에서, MAD의 비교 대상은 Poli 등에서 가져온 것이지 재학습이 아닙니다.
- **회상 과제 범위.** SWDE·SQuAD·FDA뿐이고 NQ·TriviaQA·DROP은 없습니다. MQAR과 RegBench는 그림만 있습니다.

## 9. 결론

DeltaNet을 한 줄로 요약하면 "**선형 어텐션의 덧셈 쓰기를 읽고-빼고-쓰기로 바꾼 것**"입니다. 수학적으로는 세 얼굴이 하나입니다. 온라인 회귀 $\frac{1}{2}\|Sk - v\|^2$의 SGD 한 걸음이고, 일반화 Householder $I - \beta kk^{\top}$의 곱이며, 키 방향 하나만 지우는 사영입니다.

병렬화의 열쇠는 Householder 곱이 rank-$C$라는 WY 표현과, 그 계수의 재귀가 $C \times C$ 삼각 선형계라는 UT 변환이었습니다. 결과는 "값을 $U - WS^{\top}$로 보정한 선형 어텐션"이고, 그래서 기존 청크 커널 위에 올릴 수 있었습니다.

이 규칙이 사 주는 것은 정확한 회상(MAD의 recall 계열 100, fuzzy recall 35.7)과 트랜스포머를 넘는 perplexity이고, 사 주지 않는 것은 긴 문서 추출 과제와 길이 외삽입니다. 게이트가 잊고 delta rule이 덮어쓴다는 구분은 Schlag의 부록 예제에 이미 있었고, 둘을 합치면 어떻게 되는지가 Gated DeltaNet 편입니다.

*(참고: Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021, arXiv:2102.11174 · Yang, Wang, Zhang, Shen, Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024, arXiv:2406.06484 v6 · Grazzi, Siems et al., "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues", ICLR 2025, arXiv:2411.12537 · Siems et al., "DeltaProduct", arXiv:2502.10297 · 구현은 fla-org/flash-linear-attention의 `delta_net.py`와 `ops/delta_rule/`. 2.1절의 기울기 유도, 3.2·3.3절의 귀납과 삼각계 유도, 3.4절의 복잡도 산정은 직접 한 것이고, 그림 왼쪽의 간섭 시뮬레이션($d = 64$, 무작위 단위 키, 20회 평균)도 직접 돌린 것입니다. 8절에 적은 대로 속도 수치·정밀도 분석·청크 크기 민감도는 공개되지 않았습니다.)*

#DeltaNet #delta-rule #linear-attention #fast-weights #Householder #아키텍처
