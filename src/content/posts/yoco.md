---
title: "KV 캐시를 딱 한 번만 만들면"
subtitle: "YOCO — 절반의 층이 만든 전역 key/value를 나머지 절반이 공유하는 decoder-decoder, 메모리는 1/L, prefill은 조기 종료"
description: "Transformer의 KV 캐시는 층마다 따로 쌓여 L·N·D로 자란다. Microsoft의 YOCO는 앞 절반 층(self-decoder)이 문맥을 압축해 key/value 한 벌을 만들고, 뒤 절반(cross-decoder)이 그것을 공유해 읽는다. 캐시 크기가 정확히 층 수 분의 1이 되는 이유, gated retention의 점화식, prefill을 절반에서 끊을 수 있는 구조적 이유를 유도하고, 3B 모델의 1M 문맥에서 메모리 9.4배·prefill 72배라는 숫자를 아키텍처 산수로 재구성한다. Phi-4-mini-flash, Apple 온디바이스, Gemma의 KV 공유까지 계보를 잇는다."
pubDate: 2026-09-20
order: 22
track: "추론 효율"
tags: ["KV cache","long context","cross-layer sharing","retention","prefill","추론 효율"]
hero: "/images/yoco.png"
---

GQA/MQA 편에서 KV 캐시를 줄이는 첫 번째 축을 봤습니다. **헤드 수**를 줄이는 것입니다. MLA 편은 두 번째 축, 헤드당 **차원**을 잠재 벡터로 압축하는 것이었습니다. 그런데 KV 캐시의 크기를 결정하는 인자는 하나 더 있습니다.

$$
\text{KV bytes} = \underbrace{L}_{\text{층 수}} \times N \times \underbrace{2 \cdot h_{kv} \cdot d_h \cdot b}_{\text{층당 토큰당 바이트}}
$$

$N$은 문맥 길이, $b$는 원소당 바이트입니다. 세 번째 축은 층 수 $L$, 즉 "몇 개 층이 각자 캐시를 갖는가"입니다. 표준 Transformer는 모든 층이 자기 key/value를 만들고 저장합니다. 3B급 모델이 26층이면 캐시도 26벌입니다.

Microsoft의 YOCO(You Only Cache Once)는 이 축을 건드립니다. 앞 절반 층은 캐시가 필요 없는 효율적 attention으로 문맥을 읽고, 그 출력에서 **key/value 한 벌만** 만든 뒤, 뒤 절반 층이 전부 그 한 벌을 읽습니다. 캐시가 정확히 한 벌이니 $L$ 인자가 사라지고, 비율은 층 수 그 자체가 됩니다. 그리고 이 구조에는 부산물이 하나 더 있습니다. 뒤 절반 층은 prefill 동안 **돌 필요가 없습니다.**

## 1. 구조 — self-decoder와 cross-decoder

층 수 $L$의 YOCO는 두 부분으로 나뉩니다. 입력 임베딩 $X^0 \in \mathbb{R}^{N \times D}$에서 시작해, 앞 $L/2$개 층이 **self-decoder**, 뒤 $L/2$개 층이 **cross-decoder**입니다.

$$
X^{l+1} = \mathrm{SelfDecoder}^{l}(X^{l}), \quad l = 0, \ldots, L/2 - 1
$$

self-decoder의 각 층은 causal한 **efficient self-attention**을 씁니다. 여기서 "efficient"의 조건은 하나, 디코딩 시 상태 크기가 문맥 길이와 무관하게 **상수**여야 한다는 것입니다. 논문은 두 후보를 씁니다. 기본값은 gated retention(2절)이고, 대안은 윈도우 1024의 sliding-window attention입니다. 둘 다 토큰당 상태가 $O(1)$입니다.

self-decoder의 최종 출력 $X^{L/2}$에서 전역 key/value를 **딱 한 번** 만듭니다.

$$
\hat{K} = \mathrm{LN}(X^{L/2})\, W_K, \qquad \hat{V} = \mathrm{LN}(X^{L/2})\, W_V
$$

$W_K, W_V \in \mathbb{R}^{D \times h_{kv} d_h}$는 **모델 전체에 한 쌍**입니다. RoPE는 $\hat{K}$에 적용됩니다. cross-decoder의 각 층은 자기만의 query 사영 $W_Q^l$을 갖지만 key/value는 이 $\hat{K}, \hat{V}$를 공유합니다.

$$
\hat{Q}^{l} = \mathrm{LN}(X^{l})\, W_Q^{l}, \qquad X^{l+1} = \mathrm{Attn}(\hat{Q}^{l}, \hat{K}, \hat{V}) + X^{l} \;\;(\text{이후 FFN}), \quad l = L/2, \ldots, L-1
$$

attention은 causal 마스크를 그대로 씁니다. 위치 $t$의 query는 $\hat{K}_{\le t}, \hat{V}_{\le t}$만 봅니다. 이름의 "cross"는 encoder-decoder의 cross-attention과 형태가 같다는 뜻입니다. 다만 "encoder"에 해당하는 self-decoder도 causal이므로 전체 모델은 여전히 autoregressive입니다. 그래서 논문은 이를 **decoder-decoder** 구조라 부릅니다.

## 2. gated retention — 캐시 없는 앞 절반

self-decoder가 KV 캐시를 갖지 않으려면 attention을 순환 형태로 쓸 수 있어야 합니다. 선형 attention 편에서 본 대로, softmax를 빼면 $\sum_{j \le n} (q_n \cdot k_j) v_j = q_n \left(\sum_{j \le n} k_j^{\top} v_j\right)$이고 괄호 안이 $d \times d$ 상태 $S_n$입니다. Retention은 여기에 지수 감쇠를 더합니다. gated retention(gRet)은 그 감쇠율을 **데이터 의존적으로** 만듭니다.

$$
S_n = \gamma_n S_{n-1} + K_n^{\top} V_n, \qquad O_n = Q_n S_n, \qquad \gamma_n = \sigma(X_n W_\gamma)^{1/\tau}
$$

$\gamma_n \in (0, 1)$은 토큰 $n$이 "이전 기억을 얼마나 남길지"를 결정하는 스칼라 게이트로, 헤드마다 따로 있습니다. $\sigma$는 sigmoid, $\tau$는 온도이며 공개 코드에서 16입니다. $1/\tau$ 제곱은 sigmoid 출력을 1 쪽으로 밀어 올려, 초기화 시점에 감쇠가 너무 빠르지 않게 하는 장치입니다. 예를 들어 $\sigma(\cdot) = 0.5$이면 $0.5^{1/16} \approx 0.958$입니다. Mamba-2의 스칼라 감쇠 $a_t$와 같은 자리에 있는 항이고, 이 시리즈의 SSM 편에서 정리한 "감쇠 하나로 게이팅하는 선형 순환"의 한 사례입니다.

이 점화식이 있으면 디코딩 시 상태는 헤드당 $d_h \times d_h$ 행렬 하나이고 문맥 길이에 무관합니다. 학습과 prefill에서는 길이 256 청크로 잘라 청크 안은 병렬 행렬 곱, 청크 사이는 순환으로 계산하는 chunkwise 알고리즘을 씁니다. 이 부분은 SSM 편과 linear attention 편의 내용과 동일하므로 반복하지 않겠습니다.

## 3. 캐시 산수 — 비율이 정확히 $L$인 이유

이제 위 그림의 왼쪽 패널을 유도합니다. YOCO 3B 설정은 26층, KV 헤드 8개, 헤드 차원 128, bf16입니다. 층당 토큰당 KV 바이트는

$$
2 \times 8 \times 128 \times 2 = 4096 \text{ bytes} = 4\,\mathrm{KB}
$$

같은 헤드 구성의 Transformer는 이것을 26층 모두 저장하므로 토큰당 104 KB입니다. YOCO는 cross-decoder 13개 층이 한 벌을 공유하므로 토큰당 **4 KB**, 여기에 self-decoder 13층의 gRet 상태가 더해집니다. 상태는 층당 $h \times d_h \times d_h \times b$이고, 논문 설정(헤드 24개)이면 $13 \times 24 \times 128 \times 128 \times 2 \approx 10\,\mathrm{MiB}$로 문맥 길이와 무관한 상수입니다.

$$
\frac{\text{Transformer}}{\text{YOCO}} = \frac{L \cdot N \cdot 4\,\mathrm{KB}}{N \cdot 4\,\mathrm{KB} + 10\,\mathrm{MiB}} \xrightarrow{N \to \infty} L = 26
$$

![YOCO: KV cache accounting and measured speedups](/images/yoco.png)

> 왼쪽: YOCO-3B의 설정(26층, KV 헤드 8, 헤드 차원 128, bf16)에서 계산한 시퀀스당 KV 캐시. 1M 토큰에서 Transformer 99 GiB, YOCO 3.8 GiB로 비율은 층 수 26에 수렴한다. 오른쪽: 논문이 H100에서 3B 모델로 측정한 배율. prefill 지연 시간 비율은 32K에서 2.9배, 1M에서 71.8배까지 커지고, 총 추론 메모리 비율은 9.4배, 처리량은 512K에서 9.6배다. 직접 그린 그림이다.

1M 토큰에서 계산하면 Transformer 99.2 GiB, YOCO 3.82 GiB입니다. 논문은 "1M에서 YOCO 12.4 GB, Transformer는 80 GB 초과"라고 적는데, 12.4 GB에는 KV 캐시 외에 모델 가중치(3B × 2 bytes ≈ 6 GB)와 활성값이 들어 있으므로 제 3.8 GiB와 모순되지 않습니다. 반면 Transformer의 "80 GB 초과"는 H100 한 장의 메모리를 넘어 실측이 불가능했다는 뜻이고, 제 산수로는 99 GiB + 6 GB입니다.

**총 메모리 비율이 9.4배에 그치는 이유**도 여기서 나옵니다. KV 캐시만 보면 26배지만, 분모에 가중치 6 GB가 상수로 깔려 있어 총 메모리 비율은 문맥 길이에 따라 1.95배(32K)에서 9.38배(1M)로 천천히 올라갑니다. 논문의 표 그대로이며, "KV 캐시가 1/26"과 "메모리가 1/9.4"는 다른 분모를 쓴 같은 사실입니다.

모델을 키우면 비율은 커집니다. 논문은 토큰당 KV 캐시 비율을 1.2B에서 24배, 65B에서 80배로 보고합니다. 65B 모델의 층 수가 80이므로 이 역시 "비율 = 층 수"입니다. 65B에서는 1 GB에 Transformer가 1.6K 토큰, YOCO가 128K 토큰을 담습니다.

## 4. prefill 조기 종료 — 뒤 절반은 돌 필요가 없다

이것이 YOCO의 두 번째 이득이고, 긴 문맥에서는 메모리보다 더 큰 이득입니다.

prefill의 목적은 두 가지입니다. 프롬프트의 KV 캐시를 만드는 것, 그리고 첫 토큰을 생성하는 것입니다. 표준 Transformer에서는 캐시가 모든 층에 있으므로 프롬프트 전체가 모든 층을 통과해야 합니다. YOCO에서는 캐시가 $\hat{K}, \hat{V}$ 한 벌이고, 그것은 $X^{L/2}$에서 나옵니다. 즉 **프롬프트의 마지막 토큰을 제외한 모든 위치는 self-decoder만 통과하면 캐시 생성이 끝납니다.** cross-decoder는 첫 토큰을 뽑기 위해 마지막 위치 하나에 대해서만 돌면 됩니다.

$$
\text{prefill FLOPs} \approx \underbrace{N \cdot C_{\text{self}}}_{\text{앞 절반, 전체 프롬프트}} + \underbrace{1 \cdot C_{\text{cross}}}_{\text{뒤 절반, 마지막 토큰만}}
$$

self-decoder가 gRet이므로 그 비용은 $N$에 선형입니다. Transformer의 prefill은 attention 항 때문에 $N^2$ 항을 갖습니다. 따라서 비율은 문맥 길이에 따라 **계속 커집니다.** 논문 측정값이 정확히 그렇습니다. 32K에서 2.87배, 128K에서 8.36배, 512K에서 30배, 1M에서 71.8배. 512K 프롬프트의 절대 시간은 Transformer 180초, YOCO 6초 미만입니다.

한 가지 짚을 점은, 논문의 그림에서 YOCO의 prefill 시간이 문맥 길이에 따라 거의 평평하게 약 6초 부근에 머문다는 것입니다. gRet의 비용이 선형이라면 512K에서 1M으로 갈 때 두 배가 되어야 하는데, 논문은 왜 평평한지를 설명하지 않습니다. 커널 오버헤드가 지배적이거나 측정 구간이 다를 수 있습니다. 비율 72배 자체는 논문 표의 값이지만, 그 분모의 거동은 확인하지 못했습니다.

처리량도 같은 이유로 개선됩니다. 512K에서 Transformer 4.5 tok/s, YOCO 43.1 tok/s로 9.6배입니다. 디코딩 스텝 자체는 cross-decoder가 $N$개의 $\hat{K}, \hat{V}$를 읽으므로 여전히 $O(N)$이지만, 읽는 양이 1/26이니 메모리 대역폭에 묶인 디코딩(prefill/decode 편)이 그만큼 빨라집니다.

## 5. 품질 — 무엇을 잃는가

캐시를 26벌에서 1벌로 줄였으니 표현력을 잃을 것 같습니다. 논문의 답은 "언어 모델링 손실에서는 거의 없다"입니다.

**스케일링.** 160M부터 13B까지 같은 데이터로 학습해 비교하면, 160M에서 검증 perplexity가 YOCO-gRet 3.530, YOCO-SWA 3.553, Transformer 3.564입니다. 모든 크기에서 gRet 변종이 Transformer보다 조금 낮고 SWA 변종은 Transformer와 비슷합니다. 차이가 작으므로 "동등"으로 읽는 것이 안전합니다.

**절반이라는 비율.** camera-ready 판의 ablation(160M)에서 self:cross 비율을 바꾸면 [1:1] 3.530, [3:1] 3.526, [1:3] 3.565, [0:1] 3.898입니다. self-decoder를 아예 없애면([0:1], 즉 입력 임베딩에서 바로 $\hat{K}, \hat{V}$를 만들면) 크게 나빠지고, 특히 연상 회상(AR-Hit) 항목에서 1.199 대 1.827로 무너집니다. **key/value가 문맥을 충분히 읽은 표현에서 나와야 한다**는 것이 이 구조의 전제이고, 그 "충분히"의 하한이 어딘가 1/4과 0 사이에 있습니다. [3:1]이 [1:1]보다 미세하게 좋다는 것은 self-decoder를 더 늘려도 된다는 뜻이지만, 그러면 캐시를 공유하는 층이 줄어 이득도 줄어드니 논문은 1:1을 택했습니다.

**긴 문맥.** 3B 모델을 1M까지 확장한 YOCO-3B-1M은 다중 바늘 찾기(128K)에서 바늘 수 $N = 1, 2, 4, 8$에 대해 0.98, 0.98, 0.84, 0.56입니다. 바늘이 8개일 때 LWM-7B에 집니다. 논문 초록의 "1M에서 거의 완벽한 바늘 찾기"는 바늘 하나짜리 실험이고, 바늘이 많아지면 성능이 떨어지는 것은 다른 긴 문맥 모델과 같습니다.

**반대 증거.** 비슷한 시기의 Cross-Layer Attention(CLA) 논문은 여러 KV 공유 배치를 비교하는데, 앞쪽에 KV 생성 층을 몰아 두는 "DenseFront" 배치가 perplexity 13.75로, 두 층씩 균등하게 공유하는 CLA2의 13.60보다 나빴다고 보고합니다. YOCO의 배치는 DenseFront와 형태가 같습니다. 두 논문의 self-decoder 설계가 다르므로(CLA는 self-decoder에 gRet 같은 순환 층을 두지 않음) 직접 비교는 아니지만, "앞에서 한 번만 만든 KV를 뒤가 전부 공유한다"가 항상 공짜라고 볼 수는 없다는 신호입니다.

## 6. 그 뒤 — 순환하는 self-decoder와 실제 채택

### 6.1 Universal YOCO

2026년 4월의 후속 연구(Universal YOCO, arXiv:2604.01220)는 looped transformer 편의 아이디어를 self-decoder에 결합합니다. self-decoder 블록을 $T = 3$번 반복하고, cross-decoder는 NoPE(위치 인코딩 없음), SWA 윈도우 512, 10B 총 / 1.3B 활성 MoE입니다. 측정된 디코딩 처리량(tok/s)은 다음과 같습니다.

| 문맥 | Transformer | YOCO | RINS(반복만) | YOCO-U |
|---|---|---|---|---|
| 8K | 2712 | 3356 | 1582 | 2410 |
| 256K | 137 | 318 | 56 | 303 |

KV 캐시(MB)는 8K → 256K에서 Transformer 320 → 10240, YOCO 26 → 522, YOCO-U 46 → 542입니다. 반복 구조(RINS)만 쓰면 KV 캐시가 반복 횟수만큼 늘어 처리량이 무너지는데, YOCO 구조 안에서 self-decoder만 반복하면 캐시는 여전히 한 벌이라 처리량이 유지됩니다. 품질 면에서는 80B 토큰 학습한 YOCO-U가 210B 토큰 학습한 YOCO와 비슷하다고 보고합니다. 반복이 "학습 토큰을 아낀다"는 looped transformer 편의 결론과 같은 방향입니다.

### 6.2 채택 사례

YOCO의 아이디어는 세 갈래로 흘러 들어갔습니다.

- **Phi-4-mini-flash-reasoning**(Microsoft, 2025)은 SambaY라는 변종을 씁니다. self-decoder를 Mamba+SWA 하이브리드(Samba)로 바꾸고 cross-decoder가 그 KV를 공유하는 구조로, Microsoft는 같은 크기 Phi-4-mini 대비 최대 10배 처리량을 주장합니다. YOCO가 실제 제품 모델에 들어간 첫 사례입니다.
- **Apple의 2025 온디바이스 파운데이션 모델**은 층을 두 블록으로 나누고, 뒤쪽 블록(전체의 37.5%)이 KV 사영을 갖지 않고 앞 블록의 KV를 공유합니다. 기술 보고서는 prefill 우회의 근거로 YOCO를 인용하며 첫 토큰 지연 시간 37.5% 감소를 보고합니다. 온디바이스 트랙의 "cross-layer KV 공유" 항목이 정확히 이것입니다.
- **Gemma 3n·Gemma 4**는 마지막 $N$개 층이 앞 층의 KV를 공유합니다. 다만 Gemma는 sliding-window 캐시와 global 캐시를 모두 유지하고, YOCO식 prefill 조기 종료는 하지 않습니다. 메모리 축만 가져오고 지연 시간 축은 가져오지 않은 부분 채택입니다.
- **Hunyuan-Large**는 CLA를 씁니다. 인접 두 층이 KV를 공유하는 균등 배치로, YOCO의 반대편 설계입니다.

공통점은 하나입니다. 2024년 이후 KV 캐시 축소의 세 번째 축인 "$L$"이 헤드 수·차원과 나란히 설계 변수가 되었고, 그 출발점이 YOCO입니다.

## 7. 논문이 말하지 않는 것

- **"80 GB 초과"의 측정 방법.** 1M 문맥에서 Transformer의 메모리를 어떻게 쟀는지(실측 불가라 추정인지, 다른 GPU에서 쟀는지) 명시되지 않습니다.
- **YOCO prefill이 평평한 이유.** 4절에서 적은 대로, gRet 비용이 선형이라면 나타나야 할 증가가 그림에 보이지 않습니다.
- **온도 $\tau$.** 논문 본문에는 없고 코드에서 16으로 확인됩니다. 이 값의 민감도는 보고되지 않습니다.
- **cross-decoder에서의 GQA 이득.** cross-decoder는 이미 한 벌만 캐시하므로 헤드 수 축소(GQA)와 겹치는 이득이 어디까지인지, 두 기법을 함께 쓸 때의 품질 손실이 분리되어 있지 않습니다.
- **대규모에서의 품질.** 스케일링 비교는 13B까지이고, 65B의 "80배"는 캐시 산수이지 학습된 모델의 결과가 아닙니다.
- **배치 디코딩.** 처리량 측정은 배치 크기가 명시되지 않았고, 여러 시퀀스를 배치로 묶을 때 self-decoder의 순환 상태가 어떻게 관리되는지는 다루지 않습니다.

## 8. 결론

YOCO를 한 줄로 요약하면 "**KV 캐시의 $L$ 인자를 1로 만든 구조**"입니다. 헤드 수(GQA)와 차원(MLA)에 이어 층 수라는 세 번째 축을 열었고, 비율이 정확히 층 수가 된다는 것을 3B 설정의 산수(토큰당 104 KB 대 4 KB)로 확인했습니다.

기술적으로 이 구조가 성립하는 조건은 두 가지입니다. 앞 절반이 캐시 없는 순환 attention(gRet의 $S_n = \gamma_n S_{n-1} + K_n^{\top} V_n$)이어야 상태가 상수이고, key/value가 문맥을 충분히 읽은 $X^{L/2}$에서 나와야 품질이 유지됩니다. [0:1] ablation에서 연상 회상이 무너지는 것이 두 번째 조건의 증거입니다.

그리고 부산물인 prefill 조기 종료가 긴 문맥에서는 메모리보다 큰 이득이 됩니다. 캐시가 $X^{L/2}$에서 나오니 프롬프트는 앞 절반만 통과하면 되고, 그 절반이 선형 비용이라 비율은 문맥 길이와 함께 자랍니다(1M에서 72배). Apple이 온디바이스 모델에서 가져간 것도, Phi-4-mini-flash가 10배 처리량의 근거로 든 것도 이 부분입니다. 반면 CLA 논문의 DenseFront 결과와 Gemma의 부분 채택은, 어디까지 한 벌로 줄일지가 아직 열린 설계 변수임을 보여 줍니다.

*(참고: Sun et al., "You Only Cache Once: Decoder-Decoder Architectures for Language Models", NeurIPS 2024, arXiv:2405.05254 · 코드 microsoft/unilm/YOCO. 3절의 캐시 산수(4 KB/토큰/층, 99 GiB, 비율 26)는 논문 설정에서 직접 계산한 것이며 논문의 12.4 GB·"80 GB 초과"는 가중치를 포함한 총 메모리입니다. 4·5절의 배율과 perplexity, 6.1절의 표는 각 논문의 값이고, Universal YOCO는 arXiv:2604.01220, CLA는 Brandon et al. arXiv:2405.12981, Apple은 2025 Foundation Models 기술 보고서, Phi-4-mini-flash는 Microsoft 공식 발표를 근거로 했습니다. 7절에 적은 대로 80 GB의 측정 방법·prefill 평탄 구간·배치 크기는 공개되지 않았습니다. 그림은 직접 그린 것입니다.)*

#KV-cache #long-context #cross-layer-sharing #retention #prefill #추론-효율
