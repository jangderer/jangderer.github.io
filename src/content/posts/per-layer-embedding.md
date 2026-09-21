---
title: "층마다 다른 임베딩을 토큰에게 주면"
subtitle: "Per-Layer Embedding — Gemma 3n·4의 E2B/E4B가 파라미터의 절반을 조회 테이블로 바꾼 방법, 그리고 '유효 파라미터'라는 말의 정확한 뜻"
description: "Gemma 4 E2B는 5.12B 파라미터 중 2.35B가 262,144×35×256짜리 테이블이다. 토큰 id로 층마다 256차원 벡터를 꺼내 gelu 게이트로 잔차에 섞는 Per-Layer Embedding의 정확한 수식을 HF·JAX 구현에서 읽어 내고, Google이 말하는 '유효 1.91B'가 정확히 어느 부분집합인지 체크포인트 텐서 모양으로 재현하며, 토큰당 18KB라는 조회량이 왜 테이블을 가속기 밖에 둘 수 있게 하는지 계산한다. Gemma 3n의 AltUp 가지 1~3에만 더해지던 구조가 Gemma 4에서 어떻게 바뀌었는지, Engram·STEM·MoWE와의 관계, 그리고 Google이 한 번도 보여 주지 않은 ablation까지."
pubDate: 2026-09-21
order: 23
track: "온디바이스"
tags: ["per-layer embedding","Gemma","on-device","embedding","memory","온디바이스"]
hero: "/images/ple.png"
---

온디바이스 트랙의 첫 명제는 "작은 모델일수록 어휘 임베딩이 전체를 지배한다"였습니다. 262,144개 토큰에 2048차원이면 임베딩 하나가 537M 파라미터이고, 2B 모델에서는 그것만으로 4분의 1입니다. 그래서 작은 모델은 입력과 출력 임베딩을 묶고(weight tying), 어휘를 줄이고, 차원을 낮춥니다. 임베딩을 **줄이는** 방향입니다.

Google의 Gemma 3n은 정반대로 갔습니다. 임베딩을 하나가 아니라 **층 수만큼** 둡니다. 토큰 하나가 층 $\ell$에 도달하면, 그 층 전용 테이블에서 그 토큰의 256차원 벡터를 꺼내 잔차 스트림에 섞습니다. Gemma 4 E2B에서 이 테이블은 $262{,}144 \times 35 \times 256 = 2.35\text{B}$ 파라미터로, **모델 전체의 46퍼센트**입니다. 그런데 Google은 이 모델을 "유효 2.3B"라 부릅니다. 테이블은 곱셈에 참여하지 않는 순수 조회이므로 가속기 메모리에 올릴 필요가 없다는 논리입니다.

Gemma 4 편 5절에서 이 구조를 짧게 다뤘습니다. 이 글은 그 다섯 문단을 한 편으로 펼칩니다. HF와 DeepMind JAX 구현을 줄 단위로 읽어 수식을 확정하고, "유효 파라미터"가 정확히 어느 부분집합인지 체크포인트 텐서 모양으로 재현하며, Gemma 3n에서 4로 넘어오며 바뀐 것과 바뀌지 않은 것을 가르고, 이 구조가 Engram 편의 "기억의 희소성"과 어떻게 이어지는지 봅니다. 그리고 Gemma 4 편에 잘못 적었던 세 가지도 여기서 바로잡습니다.

## 1. 파라미터는 어디에 있는가

먼저 숫자부터 고정하겠습니다. 아래는 HF에 공개된 체크포인트의 safetensors 헤더에서 텐서 모양을 읽어 합산한 값입니다. 기술 보고서의 반올림 값이 아니라 정확한 값입니다.

| | Gemma 3n E2B | Gemma 3n E4B | Gemma 4 E2B | Gemma 4 E4B |
|---|---|---|---|---|
| 모델 차원 $d$ / 층 수 $L$ | 2048 / 30 | 2048 / 35 | 1536 / 35 | 2560 / 42 |
| PLE 테이블 $262{,}144 \times L \times 256$ | 2,013M | 2,349M | 2,349M | 2,819M |
| 토큰 임베딩 (입출력 공유) | 537M | 537M | 403M | 671M |
| 나머지 텍스트 (attention·FFN·PLE 게이트) | 1,905M | 3,980M | 1,896M | 4,028M |
| 비전 + 오디오 인코더 | 983M | 983M | 476M | 478M |
| 체크포인트 합계 | 5,439M | 7,850M | 5,123M | 7,996M |
| PLE 테이블 비율 | 37.0% | 29.9% | 45.9% | 35.3% |

![PLE: parameter composition and per-token byte traffic](/images/ple.png)

> 왼쪽: 네 모델의 체크포인트 파라미터 구성. 주황색이 PLE 테이블이고, 흰 숫자는 전체 대비 비율이다. 파란 부분이 가속기에 상주해야 하는 트랜스포머 본체로, Google의 "유효 파라미터"와 대응한다. 오른쪽: Gemma 4 E2B에서 토큰 하나를 디코딩할 때 읽어야 하는 바이트 수(로그 축). 본체 가중치는 매 토큰 전부 읽지만 PLE는 토큰당 한 행, 17.9KB뿐이다. 직접 계산해 그린 그림이다.

Gemma 4 기술 보고서 Table 1은 E2B의 임베더를 "400M + 2,340M", 아인섬(einsum, 즉 attention과 FFN 행렬)을 1,870M으로 적습니다. 체크포인트의 402.7M, 2,348.8M, 1,896M과 반올림 범위에서 일치합니다. Gemma 4 편에서 "텍스트 4,610M 중 50.8%"라고 쓴 것은 이 표의 반올림 값을 나눈 것이고, 정확한 값은 텍스트 4,647M 중 **50.5퍼센트**입니다.

한 가지 눈에 띄는 것이 있습니다. Gemma 3n E2B와 Gemma 4 E2B는 모델 차원(2048 대 1536)도 층 수(30 대 35)도 다른데, 트랜스포머 본체는 1,905M과 1,896M으로 거의 같습니다. Gemma 4가 층을 더 쌓는 대신 폭을 줄여 **본체 예산을 정확히 유지**한 것입니다. 반면 PLE 테이블은 층 수에 비례하므로 2,013M에서 2,349M으로 늘었습니다. 테이블은 가속기 밖에 두는 예산이니 늘어도 된다는 판단이 읽힙니다.

## 2. 메커니즘 — 입력 쪽

기호를 정합니다. $V_p = 262{,}144 = 2^{18}$은 PLE 어휘 크기, $p = 256$은 PLE 차원, $L$은 층 수, $d$는 모델 차원입니다. $x_t$는 위치 $t$의 토큰 id, $E \in \mathbb{R}^{V \times d}$는 보통의 토큰 임베딩 테이블, $E_{\mathrm{PLE}} \in \mathbb{R}^{V_p \times Lp}$는 PLE 테이블입니다. HF 구현은 $L$개 층의 테이블을 열 방향으로 이어 붙여 하나의 2차원 테이블로 저장하고, JAX 참조 구현은 $(V_p, L, p)$ 3차원으로 둡니다. 같은 것입니다.

토큰 하나가 모델에 들어오면 두 성분이 계산됩니다.

**토큰 성분.** 테이블에서 한 행을 꺼내 $L \times p$로 재배열합니다. 보통 임베딩에 $\sqrt{d}$를 곱하듯 여기에는 $\sqrt{p} = 16$을 곱합니다.

$$
u_t = \mathrm{reshape}_{L \times p}\!\big(\sqrt{p}\; E_{\mathrm{PLE}}[x_t]\big) \in \mathbb{R}^{L \times p}
$$

**사영 성분.** 보통의 토큰 임베딩 $e_t = \sqrt{d}\, E[x_t]$를 학습되는 행렬 $W_{\mathrm{proj}} \in \mathbb{R}^{Lp \times d}$로 사영해 같은 모양으로 재배열하고, 층 슬롯마다 256차원 RMSNorm을 적용합니다.

$$
v_t = \mathrm{RMSNorm}_{p}\!\Big(\mathrm{reshape}_{L \times p}\big(d^{-1/2}\, W_{\mathrm{proj}}\, e_t\big)\Big) \in \mathbb{R}^{L \times p}
$$

$d^{-1/2}$와 $\sqrt{d}$가 상쇄되므로 실질은 $v_t = \mathrm{RMSNorm}(W_{\mathrm{proj}} E[x_t])$입니다. 이 RMSNorm의 학습 가중치는 256개 하나뿐이고 **$L$개 층 슬롯이 전부 공유**합니다. 체크포인트에서 `per_layer_projection_norm`의 텐서 크기가 정확히 256인 것으로 확인됩니다.

HF 문서는 $v_t$를 "context-aware" 성분이라 부르는데, 텍스트 경로에서는 정확한 이름이 아닙니다. $v_t$는 **현재 토큰 하나의 임베딩만의 함수**이고, attention을 거치기 전이므로 문맥은 전혀 들어 있지 않습니다. 이 성분이 진짜로 문맥을 담는 경우는 비전·오디오 인코더가 만든 소프트 토큰이 `inputs_embeds`로 들어올 때뿐입니다. 그때는 토큰 id가 없어 $u_t$를 꺼낼 수 없으므로 $v_t$만 씁니다.

**결합.** 두 성분을 더하고 $1/\sqrt{2}$를 곱합니다.

$$
g_t = \frac{1}{\sqrt{2}}\,(u_t + v_t), \qquad g^{(\ell)}_t = g_t[\ell, :] \in \mathbb{R}^{256}
$$

$1/\sqrt{2}$는 학습되지 않는 상수입니다. HF에서는 비영속 버퍼(3n) 또는 파이썬 실수(Gemma 4), JAX에서는 `jax.lax.rsqrt(2.0)`입니다. 같은 분산의 두 벡터를 더하면 분산이 두 배가 되니 그것을 되돌리는 정규화이고, 이 관례는 3n의 LAuReL 결합에서도 똑같이 쓰입니다.

한 가지 세부. PLE 어휘 $V_p = 262{,}144$는 본 어휘 $V = 262{,}400$(3n)보다 256개 작습니다. 비전과 오디오의 자리표시 토큰 256개가 빠진 것이고, 그 id가 들어오면 0번 행으로 대체해 조회합니다. Gemma 4는 본 어휘 자체가 262,144이며 멀티모달 위치는 패딩 토큰으로 바꿔 처리합니다.

## 3. 메커니즘 — 층 안에서

층 $\ell$은 자기 몫 $g^{(\ell)}_t$를 받아 잔차 스트림 $h \in \mathbb{R}^d$에 섞습니다. 두 개의 작은 행렬 $W^{(\ell)}_{\mathrm{gate}} \in \mathbb{R}^{p \times d}$, $W^{(\ell)}_{\mathrm{up}} \in \mathbb{R}^{d \times p}$와 RMSNorm 하나가 층마다 있습니다.

$$
y^{(\ell)} = \mathrm{RMSNorm}_d\!\Big(W^{(\ell)}_{\mathrm{up}}\big[\,\sigma(W^{(\ell)}_{\mathrm{gate}}\, h) \odot g^{(\ell)}_t\,\big]\Big), \qquad h \leftarrow h + y^{(\ell)}
$$

$\sigma$는 tanh 근사 GELU로, FFN이 쓰는 것과 같은 활성 함수입니다. 구조를 읽으면 이것은 **폭 256짜리 게이트 FFN**입니다. 보통의 게이트 FFN이 $W_{\mathrm{up}}[\sigma(W_{\mathrm{gate}} h) \odot (W_{\mathrm{in}} h)]$이라면, PLE는 곱해지는 쪽 $W_{\mathrm{in}} h$를 **토큰 id로 조회한 벡터 $g^{(\ell)}_t$로 바꾼 것**입니다. 은닉 상태가 "얼마나 열지"를 정하고, 토큰이 "무엇을 넣을지"를 정합니다. Engram 편의 게이트가 "주소는 토큰이, 신뢰도는 문맥이"였다면, PLE는 "내용은 토큰이, 게이트는 문맥이"입니다.

층당 추가 연산은 $2 \cdot 2 \cdot d \cdot p$ FLOPs로, 폭 $4d$짜리 게이트 FFN의 $2 \cdot 3 \cdot d \cdot 4d$에 비해

$$
\frac{4 d p}{24 d^2} = \frac{p}{6d} = \frac{256}{6 \times 1536} \approx 2.8\%
$$

입니다. 파라미터로는 층당 $2dp = 786{,}432$개(Gemma 4 E2B), 35층 합쳐 27.5M이고, 여기에 $W_{\mathrm{proj}}$ 13.8M이 더해집니다. 이 41M은 행렬 곱이므로 본체와 함께 가속기에 있어야 합니다. Gemma 4 편에서 "곱셈에 참여하지 않으니 가속기에 올릴 필요가 없다"고 쓴 것은 테이블에 대해서만 맞는 말이고, 이 41M은 예외입니다.

### 3.1 Gemma 4 — 세 번째 잔차 가지

Gemma 4 E2B/E4B의 블록은 잔차 갱신이 정확히 세 번입니다. attention, FFN, 그리고 PLE. HF `Gemma4TextDecoderLayer.forward`의 순서를 그대로 옮기면

$$
\begin{aligned}
h &\leftarrow h + \mathrm{RMSNorm}(\mathrm{Attn}(\mathrm{RMSNorm}(h))) \\
h &\leftarrow h + \mathrm{RMSNorm}(\mathrm{FFN}(\mathrm{RMSNorm}(h))) \\
h &\leftarrow h + \mathrm{RMSNorm}\!\big(W_{\mathrm{up}}[\sigma(W_{\mathrm{gate}} h) \odot g^{(\ell)}_t]\big) \\
h &\leftarrow \lambda_\ell\, h
\end{aligned}
$$

마지막 줄의 $\lambda_\ell$은 층마다 하나씩 있는 스칼라 버퍼(`layer_scalar`)로, 체크포인트에 저장된 값은 층 0에서 0.019, 층 4에서 0.49, 층 17에서 0.64, 층 34에서 0.16입니다. 보고서는 이를 "fp16에서 안정적인 추론을 위해 활성값 범위를 제한하는 스칼라"라고 설명합니다. PLE 항도 이 스칼라를 함께 곱해집니다. Gemma 4 편에서 "네 번째 잔차 가지"라고 쓴 것은 틀렸습니다. 네 번째가 되려면 LAuReL이 있어야 하는데 Gemma 4에는 없습니다.

### 3.2 Gemma 3n — AltUp 가지 1~3에만

Gemma 3n은 훨씬 복잡합니다. AltUp(Alternating Updates)이 잔차 스트림을 4개 가지 $H \in \mathbb{R}^{4 \times d}$로 유지하고, 매 층에서 가지 0(활성 가지)만 attention과 FFN을 실제로 통과시킨 뒤 나머지 세 가지는 학습된 선형 결합으로 "예측·보정"합니다. 여기에 LAuReL(rank 64의 학습된 잔차)이 attention 뒤에 하나 더 끼어듭니다. 순서는 다음과 같습니다.

1. AltUp 예측: $P = \mathrm{predict}(H)$, 활성 $a = P[0]$
2. LAuReL: $r = \tilde{a} + \mathrm{RMSNorm}(W_R W_L \tilde{a})$, $\tilde{a} = \mathrm{RMSNorm}(a)$
3. attention: $a_1 = a + \mathrm{RMSNorm}(\mathrm{Attn}(\tilde{a}))$, 결합 $a_2 = (a_1 + r)/\sqrt{2}$
4. FFN: $a_3 = a_2 + \mathrm{RMSNorm}(\mathrm{FFN}(\mathrm{RMSNorm}(a_2)))$
5. AltUp 보정: $C = \mathrm{correct}(P, a_3)$, 네 가지 모두 갱신
6. PLE: 게이트 입력은 $s = \gamma \odot C[0]$ (학습된 스케일 벡터 곱 활성 가지), $y = \mathrm{RMSNorm}(W_{\mathrm{up}}[\sigma(W_{\mathrm{gate}} s) \odot g^{(\ell)}_t])$
7. 주입: $C[i] \leftarrow C[i] + y$, **단 $i = 1, 2, 3$에 대해서만**

마지막 줄이 핵심이고, 처음 읽으면 어색합니다. 게이트는 활성 가지 0에서 계산하는데, 결과는 **활성 가지에 더하지 않고** 비활성 가지 세 개에만 더합니다. HF 코드로는 `corrected_predictions[1:] += first_prediction` 한 줄이고, JAX 참조 구현의 주석은 "outputs[0] is altup output, outputs[1:] inc PLI"입니다. 왜 이렇게 했는지에 대한 설명은 어디에도 없습니다.

결과적으로 PLE 신호는 활성 가지에 **간접적으로만** 도달합니다. 다음 층의 AltUp 예측이 네 가지의 학습된 선형 결합이므로 거기서 섞여 들어오고, 모델 끝에서는 네 가지를 각각 언임베딩해 평균하므로 거기서도 들어옵니다. 즉 3n의 PLE는 "이 층의 출력"이 아니라 "다음 층의 입력과 최종 출력"에 기여하는 구조입니다. Gemma 4가 AltUp과 LAuReL을 모두 버리면서 이 우회로도 사라졌고, PLE는 잔차 스트림에 직접 더해지게 되었습니다.

## 4. "유효 파라미터"는 정확히 무엇인가

Google은 3n E2B를 "총 5B, 유효 1.91B"라 부릅니다. 이 1.91B가 어느 부분집합인지 체크포인트에서 찾아보면

$$
\underbrace{4{,}456{,}156{,}768}_{\text{텍스트 전체}} - \underbrace{2{,}013{,}265{,}920}_{\text{PLE 테이블}} - \underbrace{537{,}395{,}200}_{\text{토큰 임베딩}} = 1{,}905{,}495{,}648
$$

로 **정확히 1.91B**입니다. 즉 3n의 "유효"는 텍스트 모델에서 PLE 테이블과 토큰 임베딩 테이블을 뺀 것이고, 비전·오디오 인코더도 뺀 것입니다. 개발자 가이드의 문장 "parameter skipping and PLE caching techniques … effective memory load of just under 2 billion (1.91B) parameters"와 맞아떨어집니다. parameter skipping이 인코더를, PLE caching이 테이블을 빼는 것입니다. 흥미롭게도 KV를 공유하는 층의 사용되지 않는 K/V 사영은 이 계산에 **포함**되어 있습니다. 체크포인트에는 있고 로드 시 버려지는 텐서인데, Google의 산정은 체크포인트 기준입니다.

Gemma 4의 "유효 2.3B / 4.5B"는 이만큼 깨끗하지 않습니다. 총 파라미터에서 PLE와 인코더를 빼면 2.30B / 4.70B, PLE와 토큰 임베딩을 빼면 2.37B / 4.51B, 보고서 Table 1의 임베더+아인섬은 2.27B / 4.61B입니다. E2B의 2.3과 E4B의 4.5를 **동시에** 재현하는 부분집합은 없습니다. "PLE 테이블을 뺀 대략의 값"으로 읽어야 하고, 정확한 공식을 주장할 수 없습니다.

다만 한 가지는 확인됩니다. 보고서 Table 3의 텍스트 전용 bf16 메모리 4.6GB(E2B), 9.0GB(E4B)는 $2 \times 2.3\text{B}$, $2 \times 4.5\text{B}$입니다. 즉 **그 표는 PLE를 뺀 유효 가중치만 센 것**입니다. PLE를 포함한 bf16 텍스트 체크포인트는 9.3GB / 15.0GB입니다. 보고서는 이 사실을 명시하지 않지만 산수가 다른 해석을 허용하지 않습니다.

## 5. 왜 테이블을 밖에 둘 수 있는가

Engram 편에서 정리한 논리가 그대로 적용됩니다. 주소가 **토큰 id만의 함수**이면 조회를 미리 발행할 수 있고, 조회량이 작으면 느린 메모리에 둬도 됩니다. PLE는 두 조건을 모두 만족합니다.

조회량을 계산하면, Gemma 4 E2B에서 토큰 하나는 테이블에서 $L \times p = 35 \times 256 = 8{,}960$개 값, bf16으로 **17.9KB**를 읽습니다. 위 그림 오른쪽이 이것을 다른 읽기와 나란히 놓은 것입니다. 본체 가중치는 매 토큰 전부 읽어야 하므로 bf16 2.3GB, 모바일 양자화로도 0.84GB입니다. 32K 문맥의 int8 KV 캐시는 53.4MB입니다. PLE 행은 그 어느 것보다 세 자릿수 이상 작습니다. 초당 20토큰이면 358KB/s이고, 스마트폰 플래시의 순차 읽기 대역폭(수 GB/s)에 비하면 무시할 수 있는 양입니다.

그래서 Google의 런타임은 테이블을 메모리에 올리지 않습니다. LiteRT-LM은 PLE를 별도의 모델 타입(`TF_LITE_PER_LAYER_EMBEDDER`)으로 취급하고 파일을 메모리 매핑해 "실제로 사용될 때만 물리 메모리를 할당"합니다. 공식 모델 카드의 숫자로 Gemma 4 E2B는 텍스트 전용 가중치 0.8GB에 임베딩 파라미터 1.12GB가 메모리 매핑되고, E4B는 2.24GB에 0.67GB입니다. Apple 모바일 CPU에서 2.58GB짜리 E2B 파일을 **물리 메모리 607MB**로 실행한다는 것이 LiteRT-LM 블로그의 수치입니다.

여기서 Gemma 4 편의 두 번째 오류를 바로잡습니다. "Google의 안드로이드 문서가 메모리 매핑된 per-layer embeddings를 명시한다"고 썼는데, 안드로이드 개발자 문서(Gemini Nano 페이지)에는 그런 언급이 없습니다. 메모리 매핑 언급은 HF의 `litert-community` 모델 카드와 Google Developers Blog의 LiteRT-LM 글에 있습니다. 그리고 "PLE를 뺀 E2B 텍스트 모델이 1GB 미만"은 Gemma 4 QAT 발표문의 문장인데, 이는 **int2/int4 모바일 양자화 모델**에 대한 것이지 bf16이 아닙니다. bf16으로는 4.6GB입니다.

### 5.1 PKM·Engram과의 자리 비교

같은 "큰 기억을 희소하게 읽는다"는 계열에서 PLE의 좌표를 찍어 보면 다음과 같습니다.

| | 주소 | 층 배치 | 게이트 | 조회 비용 |
|---|---|---|---|---|
| PKM (2019) | 은닉 상태의 query | 일부 층 | softmax top-$k$ | $O(\sqrt{M})$ key 비교 |
| PLE (Gemma 3n, 2025) | 토큰 id 하나 | **모든 층** | $\sigma(W_{\mathrm{gate}} h)$ 벡터 게이트 | $O(1)$, 층당 1행 |
| Engram (2026) | 접미사 2·3-gram 해시 | 2개 층 | $\sigma(\hat{h}^{\top}\hat{k}/\sqrt{d})$ 스칼라 | $O(1)$, 층당 16행 |

PLE는 Engram의 $n = 1$ 특수형으로 볼 수 있습니다. 해시가 필요 없는 이유는 어휘 전체를 그대로 테이블 행으로 쓰기 때문이고, 그 대가로 테이블 크기가 어휘 × 층 수 × 차원으로 고정됩니다. Engram이 슬롯 수를 자유롭게 키우고 n-gram으로 문맥을 주소에 넣는 대신 층 두 개에만 두는 것과 정확히 반대의 선택입니다. Engram 논문은 관련 연구에서 PLE를 "embedding scaling" 계열로 명시적으로 인용하며, 자신은 "합성적 n-gram 구조를 표현 공간에 직접 통합하는" 별개 계열이라고 선을 긋습니다.

## 6. 3n에서 4로 — 무엇이 남고 무엇이 사라졌나

| | Gemma 3n | Gemma 4 |
|---|---|---|
| 테이블 기하 ($V_p$, $p$) | 262,144 / 256 | 동일 |
| 입력 결합 $(u + v)/\sqrt{2}$ | 동일 | 동일 |
| 층 내 게이트 $\sigma(W_{\mathrm{gate}} \cdot) \odot g$ | 동일 (GELU tanh) | 동일 |
| 게이트 입력 | $\gamma \odot C[0]$ (AltUp 보정 활성 가지) | 잔차 $h$ 그대로 |
| 주입 대상 | AltUp 가지 1~3만 | 단일 잔차 스트림 |
| AltUp / LAuReL / 활성 희소성 | 있음 | **없음** |
| 층 스칼라 $\lambda_\ell$ | 없음 | 있음 (fp16 범위 제한) |
| E2B와 E4B의 관계 | E2B는 E4B의 MatFormer 부분 모델 (35층 중 30층, FFN 8192) | 별개 체크포인트 ($d$, $L$, KV 헤드 모두 다름) |
| 12B 이상 | 해당 없음 | PLE 없음 (`hidden_size_per_layer_input = 0`) |

PLE 자체는 거의 그대로이고, 그 주변이 정리되었습니다. 3n의 AltUp·LAuReL·활성 희소성은 모두 "작은 연산으로 큰 모델처럼"이라는 목표의 실험적 장치였고, Gemma 4는 그중 PLE와 KV 공유만 남겼습니다. 어떤 것이 효과가 있었는지에 대한 ablation은 어느 보고서에도 없지만, 무엇이 살아남았는지가 간접적 답입니다.

3n에서 E2B가 E4B의 부분 모델이라는 것은 개발자 가이드의 "E4B의 MatFormer 학습 중 E2B 부분 모델을 동시에 최적화"라는 문장과, HF 메타데이터의 `base_model: google/gemma-3n-E4B`, 그리고 두 모델의 인코더 파라미터가 바이트 단위로 같다는 사실(983,281,504)로 확인됩니다. 35층 중 어느 5개를 뺐는지, E2B의 PLE 테이블이 E4B 테이블의 30개 슬라이스를 그대로 쓴 것인지는 공개되지 않았습니다. Gemma 4에서는 이 관계가 사라졌고, Gemma 4 편 8절에 적은 대로 MatFormer도 쓰지 않습니다. 다만 그 절에서 "Gemma 3n의 내용이 잘못 옮겨진 것"이라고 한 범위는 MatFormer·AltUp·LAuReL에 한정해야 합니다. PLE는 보고서가 "Gemma 3n에서와 같이" 쓴다고 명시합니다.

## 7. 표현력 — 무엇을 얻는가

Google이 PLE의 효과에 대해 말하는 것은 "가속기 메모리를 늘리지 않고 품질을 크게 높인다"(개발자 가이드)와 "층을 더 쌓거나 파라미터를 더하는 대신 각 디코더 층에 토큰마다 작은 임베딩을 준다"(Gemma 4 모델 카드)가 전부입니다. **수치 ablation은 없습니다.** 그래서 여기서는 구조에서 읽을 수 있는 것만 적겠습니다.

3절의 식 $y = W_{\mathrm{up}}[\sigma(W_{\mathrm{gate}} h) \odot g_t]$에서 $g_t$를 고정하면 $y$는 $h$의 함수이고, $h$를 고정하면 $y$는 $g_t$의 선형 함수입니다. 즉 PLE는 층마다 **토큰이 결정하는 rank-256 방향으로의 이동을, 문맥이 결정하는 게이트로 크기 조절**해 더하는 것입니다. 보통의 트랜스포머에서 토큰의 정체성은 층 0의 임베딩 한 번으로만 주입되고, 이후 층은 잔차 스트림에 남은 흔적을 통해서만 그것을 알 수 있습니다. PLE는 모든 층에 "지금 처리 중인 토큰이 무엇인지"를 **직접** 다시 알려 줍니다. 깊은 층에서 토큰 정체성이 희미해지는 문제에 대한 파라미터적 답이고, 그 파라미터가 조회 테이블이라 싸다는 것이 요점입니다.

STEM(Sadhukhan et al., 2026)은 이 관찰을 밀고 나가 FFN의 up-projection 자체를 토큰 인덱스 조회로 바꾸는데, 그 논문의 PLE 서술이 유용합니다. "PLE는 전문가 부분망 사이에 FFN의 게이트 사영과 다운 사영을 공유하면서, 기존 FFN을 없애지 않고 그 옆에 추가 블록으로 붙인다." 즉 PLE는 **어휘 크기만큼의 전문가를 가진 MoE에서, 전문가마다 다른 것은 256차원 벡터 하나뿐**인 극단적으로 가벼운 형태로 볼 수 있습니다. MoWE(Mixture of Word Experts, 2023)가 단어별 전문가를 희소 기억으로 쓴 것의 후손입니다.

## 8. 공개되지 않은 것

- **ablation.** PLE 유무에 따른 품질 차이를 Google은 어디에도 싣지 않았습니다. "유효 2B 모델이 2B dense보다 낫다"는 주장도 없습니다. 3n의 기술 보고서 자체가 나오지 않았고(2025년 5월 발표문이 "곧 기술 보고서"라 했으나 Gemma 4 보고서조차 3n을 웹 페이지로만 인용합니다), Gemma 4 보고서는 PLE를 두 문장으로 언급합니다.
- **테이블 학습.** 학습률, weight decay, 초기화 분포 모두 미공개입니다. Engram이 테이블에 5배 학습률과 weight decay 0을 썼다고 밝힌 것과 대조됩니다. 공개 구현의 초기화(JAX `normal()`, HF `initializer_range=0.02`)는 추론 코드의 것이지 학습 시의 것이라는 보장이 없습니다.
- **온디바이스 양자화 비트 수.** 모바일 포맷이 "int2와 int4 혼합"이라는 것은 있으나 PLE 테이블의 비트 수는 없습니다. E4B 모델 카드의 "임베딩 파라미터 0.67GB"를 테이블 2.82B개로 나누면 약 1.9비트로 int2에 맞지만, E2B의 1.12GB는 2.35B개에 대해 3.8비트로 맞지 않습니다. 두 값에 다른 것이 포함되어 있을 가능성이 크고, 비트 수를 단정할 수 없습니다.
- **조회 지연 시간.** 메모리 매핑된 테이블에서 행을 읽는 실제 지연은 측정값이 없습니다. 모델 카드에는 전체 TTFT와 디코딩 속도만 있습니다.
- **가지 1~3에만 더하는 이유.** 3n의 이 설계에 대한 설명은 코드 주석 한 줄뿐입니다.
- **공유 RMSNorm.** 사영 성분의 256차원 RMSNorm을 층마다 두지 않고 하나로 공유하는 이유, 그리고 PLE 어휘에서 멀티모달 토큰 256개를 뺀 이유는 설명되지 않습니다.
- **PLE 테이블의 메모리 매핑은 보고서의 주장이 아닙니다.** Gemma 4 기술 보고서는 PLE가 플래시나 CPU에 있다고 어디에도 쓰지 않습니다. 그 주장은 3n 개요 페이지("빠른 로컬 저장소에 캐시")와 LiteRT-LM 자료에만 있습니다.

## 9. 결론

PLE를 한 줄로 요약하면 "**층마다 하나씩, 토큰 id로 조회하는 256차원 게이트 FFN 입력**"입니다. 수식으로는 $g_t = (u_t + v_t)/\sqrt{2}$가 입력이고 $h \leftarrow h + \mathrm{RMSNorm}(W_{\mathrm{up}}[\sigma(W_{\mathrm{gate}} h) \odot g^{(\ell)}_t])$가 층의 갱신이며, 그 사이의 모든 스케일($\sqrt{p}$, $d^{-1/2}$, $1/\sqrt{2}$)은 상수입니다.

이 구조가 온디바이스에서 의미 있는 이유는 파라미터 수가 아니라 **메모리 계층에서의 자리**입니다. 테이블은 토큰 id만으로 주소가 정해지고 토큰당 18KB만 읽으므로 가속기 밖, 심지어 플래시에 둘 수 있습니다. 3n E2B의 "유효 1.91B"가 텍스트에서 PLE 테이블과 토큰 임베딩을 뺀 값과 파라미터 단위로 일치한다는 것, 그리고 Gemma 4 E2B의 본체가 3n E2B와 거의 같은 1.9B로 유지되면서 테이블만 2.35B로 늘어난 것이 그 설계 의도의 증거입니다.

그리고 Gemma 3n에서 4로 넘어오며 AltUp도 LAuReL도 MatFormer도 사라졌지만 PLE는 남았습니다. Google이 효과를 수치로 보여 준 적은 없지만, 두 세대에 걸쳐 살아남은 것이 그들 나름의 답입니다. 이 구조를 n-gram과 해시로 일반화한 것이 Engram이고, FFN 자체를 조회로 바꾼 것이 STEM입니다. "기억의 희소성"이라는 축은 이제 온디바이스와 프런티어 양쪽에서 동시에 열리고 있습니다.

*(참고: 구현은 HuggingFace transformers `models/gemma3n/modeling_gemma3n.py`와 `models/gemma4/modeling_gemma4.py`, DeepMind 공식 JAX 참조 구현 `google-deepmind/gemma`의 `gemma3n/_modules.py`·`gemma4/_modules.py`에서 읽었습니다. 파라미터 수는 HF 체크포인트의 safetensors 헤더에서 텐서 모양을 합산한 것이며(3n은 바이트 동일한 미러 사용, 합계가 Google HF API의 값과 일치), Gemma 4 기술 보고서(arXiv:2607.02770) Table 1·3, Gemma 3n 개발자 가이드·모델 카드, Gemma 4 모델 카드·QAT 발표, LiteRT-LM 블로그와 `litert-community` 모델 카드를 인용했습니다. 3절의 FLOPs 비율과 5절의 바이트 계산은 직접 한 것입니다. Gemma 4 편 5절의 "네 번째 잔차 가지", "안드로이드 문서", "1GB 미만"의 세 부분은 이 글에 맞춰 수정했습니다. 그림은 직접 그린 것입니다.)*

#per-layer-embedding #Gemma #on-device #embedding #memory #온디바이스
