---
title: "어텐션 없이 기억하는 법"
subtitle: "SSM에서 Mamba-2까지 — 제어이론에서 출발한 RNN이 어텐션과 같은 행렬에 도착하다"
description: "연속 시간 상태공간 모델을 이산화하면 RNN이자 convolution이 된다. Mamba는 여기에 입력 의존(selective) 파라미터를, Mamba-2(SSD)는 스칼라 감쇠를 넣어 어텐션과 같은 반분리 행렬로 통합한다. 유도·복잡도·한계까지."
pubDate: 2026-09-03
order: 6
track: "아키텍처"
tags: ["SSM","Mamba","Mamba-2","SSD","아키텍처"]
hero: "/images/ssm_ssd.png"
---

linear attention 편에서 우리는 어텐션의 커널을 바꾸면 **RNN처럼 고정 크기 상태로 디코딩**할 수 있다는 걸 봤습니다. 그런데 같은 목적지에 **전혀 다른 길로** 도착한 계열이 있습니다. 제어이론의 **state space model(SSM)** 에서 출발해, S4를 거쳐, Mamba와 Mamba-2로 이어지는 흐름입니다.

이 글은 그 길을 수식으로 따라갑니다. 연속 시간 미분방정식이 어떻게 RNN이 되고, 왜 동시에 convolution이 되며, Mamba가 무엇을 바꿨고, Mamba-2가 왜 "어텐션과 같은 것"이라고 말하는지까지요. 기호는 등장할 때마다 정의하고, 용어는 영어를 그대로 씁니다.

## 1. 출발점: 연속 시간 상태공간 모델

SSM은 원래 신호 처리·제어이론의 도구입니다. 스칼라 입력 신호 $x(t)\in\mathbb{R}$이 들어오면, **숨은 상태** $h(t)\in\mathbb{R}^{N}$을 갱신하고, 그 상태에서 출력 $y(t)\in\mathbb{R}$을 읽습니다.

$$
h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t).
$$

기호를 하나씩 봅시다. $t$는 연속 시간, $h'(t)$는 상태의 시간 미분(상태가 지금 어느 방향으로 변하는지), $N$은 **상태 차원**(state size — 얼마나 많은 정보를 기억할 수 있는지를 정하는 크기)입니다. $A\in\mathbb{R}^{N\times N}$은 상태가 스스로 어떻게 변하는지(**감쇠·회전**)를 정하는 행렬, $B\in\mathbb{R}^{N\times 1}$은 입력을 상태에 **써넣는** 방향, $C\in\mathbb{R}^{1\times N}$은 상태에서 출력을 **읽어내는** 방향입니다.

쉽게 말해, $B$는 "듣기", $A$는 "기억이 흐려지는 방식", $C$는 "말하기"입니다. 언어 모델에서는 이 스칼라 SSM을 임베딩 차원 $d$개의 채널마다 하나씩 독립적으로 둡니다(채널 $d$개 × 상태 $N$ = 층당 상태 $dN$).

## 2. 이산화 — 미분방정식이 RNN이 되다

토큰은 연속 신호가 아니라 이산 열 $x_1,x_2,\dots,x_L$입니다. 그래서 위 식을 **간격 $\Delta$** 로 이산화합니다. 가장 흔한 방법이 zero-order hold(ZOH: 한 스텝 동안 입력이 상수라고 가정)로, 결과는

$$
\bar A = e^{\Delta A}, \qquad
\bar B = (\Delta A)^{-1}\big(e^{\Delta A}-I\big)\,\Delta B .
$$

$\Delta>0$는 **스텝 크기**(한 토큰이 "몇 초"에 해당하는지), $\bar A,\bar B$는 그 스텝 크기에 맞게 변환된 이산 파라미터입니다. Mamba 계열은 $\bar B\approx \Delta B$로 단순화(Euler 근사)하기도 합니다. 이제 상태 갱신은 그냥 **RNN**입니다.

$$
h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t .
$$

여기서 $h_t\in\mathbb{R}^N$은 토큰 $t$까지 읽은 뒤의 상태, $x_t$는 $t$번째 입력, $y_t$는 $t$번째 출력입니다. 디코딩 때는 토큰 하나당 $O(N)$ 연산(채널 $d$개면 $O(dN)$)에, 저장은 $h_t$ 하나뿐. **컨텍스트가 100만 토큰이어도 상태 크기는 그대로**입니다.

## 3. 같은 식이 convolution이기도 하다 — 유도

RNN은 학습이 느립니다(토큰을 순서대로 처리해야 하니 병렬화가 안 됨). 그런데 $\bar A,\bar B,C$가 **시간에 무관하게 고정**이면(linear time-invariant, LTI), 재귀를 풀어 쓸 수 있습니다. $h_0=0$에서 시작해 한 줄씩 전개하면

$$
\begin{aligned}
h_1 &= \bar B x_1,\\
h_2 &= \bar A\,\bar B x_1 + \bar B x_2,\\
h_3 &= \bar A^2\bar B x_1 + \bar A\bar B x_2 + \bar B x_3,\\
&\ \vdots\\
h_t &= \sum_{j=1}^{t} \bar A^{\,t-j}\,\bar B\,x_j .
\end{aligned}
$$

여기에 $C$를 곱하면 출력은

$$
y_t = \sum_{j=1}^{t} \underbrace{C\,\bar A^{\,t-j}\bar B}_{\displaystyle \bar K_{t-j}}\;x_j
\;=\; (\bar K * x)_t ,
$$

즉 **커널** $\bar K=(C\bar B,\ C\bar A\bar B,\ C\bar A^2\bar B,\dots)$ 와 입력의 **causal convolution**입니다. $\bar K_{m}$은 "$m$스텝 전 입력이 지금 출력에 미치는 영향"이고, $\bar A^{m}$이 그 영향의 감쇠를 정합니다. convolution은 FFT로 $O(L\log L)$에 계산되니, **학습은 convolution(병렬), 추론은 RNN(상수 상태)** — 이것이 S4가 확립한 이중 관점입니다.

단, 조건이 있었죠. $\bar A,\bar B,C$가 **토큰에 따라 변하면 안 됩니다.** 그래서 LTI SSM은 "어떤 토큰이 중요한지 골라서 기억"할 수 없습니다. 모든 입력을 같은 규칙으로 흐려지게 기억할 뿐이죠. 어텐션이 잘하는 **내용 기반 선택**이 빠져 있습니다.

## 4. Mamba — 파라미터를 입력에 의존시키다 (selective SSM)

Mamba(Gu & Dao, 2023)의 한 줄 아이디어: **$\Delta, B, C$를 입력 $x_t$의 함수로 만들자.**

$$
\Delta_t = \mathrm{softplus}(W_\Delta x_t),\qquad B_t = W_B x_t,\qquad C_t = W_C x_t .
$$

$W_\Delta, W_B, W_C$는 학습되는 선형 사상이고, softplus는 $\Delta_t>0$를 보장합니다. $A$는 대각 행렬(S4D 방식)로 두어 $\bar A_t = e^{\Delta_t A}$가 원소별 지수로 싸게 계산됩니다. 그러면 갱신식은

$$
h_t = \bar A_t\,h_{t-1} + \bar B_t\,x_t,\qquad y_t = C_t\,h_t ,
$$

이제 $\Delta_t$가 크면 $\bar A_t=e^{\Delta_t A}$가 0에 가까워져 **이전 기억을 지우고 현재 입력을 크게 씁니다**(reset). $\Delta_t$가 작으면 $\bar A_t\approx I$라 **이전 기억을 보존하고 현재 입력은 무시**합니다(skip). 쉽게 말해, $\Delta_t$가 RNN의 forget/input 게이트 역할을 **하나의 스칼라**로 합니다.

대가는 §3의 convolution 관점을 잃는다는 것입니다(커널이 토큰마다 달라지니까). Mamba는 대신 **parallel scan**으로 학습을 병렬화합니다. 재귀 $h_t=a_t h_{t-1}+b_t$($a_t=\bar A_t$, $b_t=\bar B_t x_t$)는 쌍 $(a,b)$에 대한 **결합법칙을 만족하는 연산**

$$
(a_1,b_1)\oplus(a_2,b_2) = (a_1a_2,\ a_2 b_1 + b_2)
$$

의 누적입니다(연속 두 스텝을 합치면 감쇠는 곱해지고, 입력은 뒤의 감쇠만큼 줄어든 채 더해진다는 뜻). 결합법칙이 성립하면 트리 구조로 $O(\log L)$ 깊이에 병렬 계산할 수 있습니다. 여기에 FlashAttention과 같은 정신으로 **상태 $h$를 HBM에 쓰지 않고 SRAM 안에서 scan을 끝내는** 하드웨어 인지 커널을 붙인 것이 Mamba입니다.

## 5. Mamba-2 — SSM은 사실 어텐션과 같은 행렬이다 (SSD)

Mamba-2(Dao & Gu, 2024)는 제약을 하나 더 겁니다. **$A_t$를 스칼라 곱하기 단위행렬로 두자**: $\bar A_t = a_t I$, $a_t\in(0,1)$. 즉 상태의 모든 차원이 **같은 비율로** 흐려집니다. 그러면 §3의 전개를 selective 버전에 그대로 적용할 수 있습니다.

$$
y_t = \sum_{j=1}^{t} C_t^{\top}\Big(\prod_{k=j+1}^{t} a_k\Big)B_j\,x_j
\;=\; \sum_{j=1}^{t} M_{tj}\,x_j,
\qquad
M_{tj} = \underbrace{\Big(\prod_{k=j+1}^{t} a_k\Big)}_{L_{tj}}\;C_t^{\top}B_j .
$$

$\prod_{k=j+1}^{t}a_k$는 "토큰 $j$의 입력이 $t$까지 오는 동안 누적된 감쇠"이고, $j>t$면 $M_{tj}=0$(미래는 안 봄)입니다. 이제 이 식을 어텐션과 나란히 놓아 보세요.

$$
\text{attention: } y_t=\sum_{j\le t}\underbrace{\mathrm{softmax}(\cdot)_{tj}}_{\text{가중치}}\,v_j,\qquad
\text{SSD: } y_t=\sum_{j\le t}\underbrace{L_{tj}\,C_t^{\top}B_j}_{\text{가중치}}\,x_j .
$$

$C_t$가 query, $B_j$가 key, $x_j$가 value, $L_{tj}$가 **감쇠 마스크** 역할입니다. 특히 $a_k\equiv 1$이면 $L_{tj}=1$이 되어 **정규화 없는 causal linear attention과 정확히 같아집니다.** 즉 SSD(structured state space duality)는 "selective SSM = 감쇠 마스크가 붙은 linear attention"이라는 등식입니다. 행렬 $M$은 아래 삼각이고 각 부분행렬의 랭크가 $N$ 이하인 **반분리(semiseparable) 행렬**입니다.

![SSD 행렬과 청크 분해](/images/ssm_ssd.png)
> 왼쪽: SSD 행렬 $M_{tj}=C_t^{\top}(\prod_{k=j+1}^{t}a_k)B_j$. 아래 삼각이고, 대각선에서 멀어질수록 누적 감쇠로 값이 작아집니다. 빨간 실선 대각 블록은 청크 안의 토큰끼리 — 어텐션처럼 행렬곱으로 계산합니다. 초록 점선의 비대각 블록은 랭크가 $N$ 이하라서, 명시적으로 만들지 않고 **청크 경계의 상태 하나**로 전달합니다. 오른쪽: 디코딩 때 GPU에 상주해야 하는 메모리. Transformer의 KV cache는 컨텍스트에 비례해 커지지만(Llama-3-70B, GQA-8 기준 1M 토큰에 ~330GB), SSM 상태는 컨텍스트와 무관하게 상수입니다.

이 등식이 중요한 이유는 **알고리즘**입니다. 시퀀스를 길이 $Q$의 청크로 자르면

- **청크 안**(대각 블록): $Y_{\text{intra}} = (L_c \circ C_cB_c^{\top})X_c$ — 크기 $Q\times Q$의 작은 어텐션이니 **행렬곱(tensor core)** 으로 계산.
- **청크 사이**(비대각 블록): 청크 끝의 상태 $h_c=\sum_{j\in c}(\prod_{k>j}a_k)\,B_j x_j^{\top}\in\mathbb{R}^{N\times P}$($P$는 헤드 차원)만 다음 청크로 넘기고, 다음 청크는 $C_t^{\top}(\prod a)\,h_{c}$로 기여분을 받습니다 — **재귀**.

여기서 $\circ$는 원소별 곱, $X_c\in\mathbb{R}^{Q\times P}$는 청크 $c$의 입력, $B_c,C_c\in\mathbb{R}^{Q\times N}$은 그 청크의 key·query에 해당합니다. Mamba의 scan은 원소별 연산이라 tensor core를 못 썼는데, SSD는 대부분을 행렬곱으로 바꿔 **학습 속도가 Mamba 대비 2~8배** 빨라집니다(논문 수치). 또 헤드 구조(multi-head, $P=64$, $N=128$ 등)와 tensor parallel 같은 Transformer의 시스템 기법을 그대로 가져올 수 있게 됐습니다.

## 6. 복잡도 정리

$L$은 시퀀스 길이, $d$는 모델 차원, $N$은 상태 크기(보통 $N\ll L$).

| | 학습(토큰 $L$개) | 디코딩(토큰 1개) | 디코딩 시 상주 메모리 |
|---|---|---|---|
| Softmax attention | $O(L^2 d)$ | $O(Ld)$ | KV cache $O(Ld)$ |
| Linear attention | $O(L d^2)$ | $O(d^2)$ | 상태 $O(d^2)$ |
| Mamba (scan) | $O(L d N)$ (원소별) | $O(dN)$ | 상태 $O(dN)$ |
| Mamba-2 (SSD) | $O(L d N)$ + 청크 행렬곱 | $O(dN)$ | 상태 $O(dN)$ |

디코딩에서 attention만 $L$에 비례합니다. 컨텍스트가 길수록 SSM 쪽이 유리해지는 이유가 이 한 줄에 있습니다.

## 7. 한계 — 고정 크기 상태의 대가

그러나 공짜는 아닙니다. 상태가 $dN$개의 숫자로 **고정**이라는 말은, 컨텍스트에서 **임의의 토큰을 정확히 꺼내 오는 일**(exact recall, copying)에 근본적 상한이 있다는 뜻입니다. 어텐션은 $L$개의 key를 모두 보관하니 어떤 토큰이든 찾아올 수 있지만, SSM은 "요약본"만 들고 있습니다. 실제로 전화번호부 조회(phonebook lookup)나 긴 문자열 복사에서 순수 Mamba 계열이 Transformer에 뒤처진다는 보고가 반복됩니다(Jelassi et al., 2024; Waleffe et al., 2024). in-context learning의 일부 능력도 같은 이유로 약해집니다.

이 약점을 **어텐션 층 몇 개**로 메우는 것이 다음 편의 **hybrid 아키텍처**입니다.

## 8. 결론

SSM은 "미분방정식 → 이산화 → RNN이자 convolution"이라는 정갈한 수학에서 출발해, Mamba에서 **입력 의존 게이트**를 얻고, Mamba-2에서 **어텐션과 같은 반분리 행렬**로 통합됐습니다. linear attention이 어텐션 쪽에서 RNN을 향해 걸어왔다면, SSM은 RNN 쪽에서 어텐션을 향해 걸어와 **같은 행렬에서 만난** 셈입니다. 남은 문제는 하나 — 고정 크기 상태가 기억하지 못하는 것을 무엇으로 보완할 것인가.

*(참고: S4 — Gu et al., arXiv:2111.00396 · Mamba — Gu & Dao, arXiv:2312.00752 · Mamba-2/SSD — Dao & Gu, arXiv:2405.21060 · recall 한계 — Jelassi et al. "Repeat After Me", arXiv:2402.01032; Waleffe et al., arXiv:2406.07887. 그림은 모두 직접 그린 것이며 수치는 위 설정(층 수·차원)에서의 계산값입니다.)*

#SSM #Mamba #Mamba2 #SSD #아키텍처
