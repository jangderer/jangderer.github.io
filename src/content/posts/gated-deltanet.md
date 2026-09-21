---
title: "잊는 게이트와 덮어쓰는 규칙을 한 식에"
subtitle: "Gated DeltaNet — Mamba-2의 스칼라 감쇠 α와 DeltaNet의 delta rule β를 곱한 전이 α(I−βkkᵀ), 게이트가 WY 표현에 접히는 방식, 그리고 Qwen3-Next·Kimi Linear가 3:1 하이브리드로 채택한 이유"
description: "Mamba-2의 게이트는 모든 연관을 같은 비율로 잊고, DeltaNet의 delta rule은 키 하나만 덮어쓰되 지우지 못한다. Gated DeltaNet은 전이를 α_t(I−β_t k_t k_tᵀ)로 두어 둘을 합친다. α를 Mamba-2식 exp(−exp(A)·softplus)로 두는 코드 수준의 매개화, 스칼라 게이트가 Householder 곱과 가환이라 WY 표현이 γ만 곱해 그대로 서는 유도, S-NIAH 세 과제에서 각 규칙이 무너지는 자리, 1.3B/100B 결과의 실제 격차(Mamba-2 대비 +0.43), 그리고 Qwen3-Next의 config에서 GDN 층 상태 1 MiB 대 attention 층 KV 2 KiB/토큰을 계산해 3:1 하이브리드의 메모리를 재구성한다. Kimi Linear의 KDA가 게이트를 채널별로 바꾼 이유까지."
pubDate: 2026-09-21
order: 26
track: "아키텍처"
tags: ["Gated DeltaNet","delta rule","gating","Qwen3-Next","Kimi Linear","linear attention","아키텍처"]
hero: "/images/gdn.png"
---

DeltaNet 편의 2.3절에서 Schlag의 부록 예제 하나를 봤습니다. 정규직교 키 $k_1, k_2$로 두 값을 저장한 뒤 $k_2$로 새 값을 쓸 때, 게이트 규칙은 무관한 $v_1$까지 $(1 - \beta)$배로 감쇠시키고 delta rule은 $v_1$을 그대로 둡니다. 그때의 결론은 "게이트는 잊고, delta rule은 덮어쓴다"였습니다.

그런데 잊는 것이 필요할 때가 있습니다. 문맥이 바뀌어 지금까지 저장한 연관이 전부 쓸모없어졌다면, 키 하나씩 덮어쓰는 것으로는 느립니다. DeltaNet은 상태를 통째로 비우는 수단이 없습니다. 반대로 Mamba-2는 통째로 비우는 것밖에 못 합니다. Yang·Kautz·Hatamizadeh(2024)의 Gated DeltaNet은 이 둘을 한 식에 넣습니다.

$$
S_t = S_{t-1}\,\alpha_t\big(I - \beta_t k_t k_t^{\top}\big) + \beta_t v_t k_t^{\top}
$$

식 자체는 DeltaNet에 스칼라 $\alpha_t$ 하나를 곱한 것입니다. 이 글은 그 스칼라가 무엇을 바꾸는지를 봅니다. 매개화를 코드에서 확정하고, 청크 알고리즘에 게이트가 어떻게 접히는지 유도하며, 합성 과제에서 세 규칙이 각각 어디서 무너지는지, 실제 언어 모델에서 격차가 얼마인지를 봅니다. 그리고 이 층이 2025년 9월 이후 Qwen3-Next, Qwen3.5, Kimi Linear의 선형 층이 된 과정을 config 수준에서 정리합니다.

## 1. 두 축 — 잊기와 덮어쓰기

Linear attention 편의 재귀 $S_t = S_{t-1} + v_t k_t^{\top}$에서 출발해, 이 시리즈가 지나온 두 갈래를 나란히 놓습니다.

$$
\text{Mamba-2:}\quad S_t = \alpha_t S_{t-1} + v_t k_t^{\top}, \qquad
\text{DeltaNet:}\quad S_t = S_{t-1}(I - \beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}
$$

$\alpha_t \in (0, 1)$은 토큰이 정하는 스칼라 감쇠이고, $\beta_t \in (0, 1)$은 쓰기 강도입니다. 논문의 표현으로 Mamba-2의 게이트는 "매 스텝 모든 키-값 연관을 균일하게 감쇠시킨다. 특정 연관 하나를 잊어야 한다면 모든 연관이 똑같이 잊힌다." DeltaNet은 "한 번에 키-값 쌍 하나만 수정하므로, 특히 문맥 전환에서 낡은 정보를 빠르게 비울 수 없다."

Gated DeltaNet의 전이는 두 행렬의 곱입니다. $\alpha_t \to 0$이면 상태가 통째로 비워지고, $\alpha_t \to 1$이면 순수 delta rule로 돌아갑니다. 온라인 학습 관점(DeltaNet 편 2.1절)에서는 delta rule이 손실 $\frac{1}{2}\|Sk_t - v_t\|^2$의 SGD 한 걸음이었는데, 여기에 $\alpha_t$를 곱하는 것은 **적응적 weight decay**를 넣는 것입니다. 갱신 전에 $S$를 $\alpha_t$배로 줄이고 나서 SGD를 밟습니다.

전이의 스펙트럼도 그대로 따라옵니다. $\|k_t\|_2 = 1$이면 $\alpha_t(I - \beta_t k_t k_t^{\top})$의 고유값은 $k_t$ 방향으로 $\alpha_t(1 - \beta_t)$, 나머지 $d - 1$개 방향으로 $\alpha_t$입니다. 모두 $(0, 1)$ 안이고, DeltaNet 편 그림 오른쪽의 주황 점선이 이것입니다.

## 2. 매개화 — 코드에서 확정한 것

논문은 "$\alpha$에 Mamba-2의 매개화를 쓰되 간결함을 위해 생략한다"고만 적습니다. 공개 구현(flash-linear-attention의 `gated_deltanet.py`, NVIDIA의 학습 코드, HF의 Qwen3-Next)이 일치하므로 거기서 확정합니다. 값 헤드 $h$, 토큰 $t$에 대해

$$
g_t = -\exp(A_{\log,h})\cdot\mathrm{softplus}\big(a_t + b_{\mathrm{dt},h}\big) \le 0, \qquad \alpha_t = \exp(g_t) \in (0, 1)
$$

$a_t = (W_a x_t)_h$는 헤드당 스칼라 하나로 사영된 값이고, $A_{\log,h}$와 $b_{\mathrm{dt},h}$는 헤드당 학습 파라미터입니다. Mamba 편의 $\bar{A} = \exp(\Delta A)$에서 $A = -\exp(A_{\log})$, $\Delta = \mathrm{softplus}(\cdot + b_{\mathrm{dt}})$인 것과 정확히 같은 구조이고, 초기화도 같습니다. $A_{\log}$는 $\log\mathrm{Uniform}(0, 16)$, $b_{\mathrm{dt}}$는 $\Delta \sim \mathrm{LogUniform}(0.001, 0.1)$의 softplus 역함수. 둘 다 weight decay를 받지 않습니다. $\beta_t = \sigma(W_b x_t)_h$는 헤드당 sigmoid입니다.

나머지는 DeltaNet 편 4절과 같습니다. query·key는 커널 4의 짧은 conv → SiLU → L2 정규화, value는 conv → SiLU, 출력은 헤드별 RMSNorm에 $\mathrm{SiLU}(W_g x_t)$를 곱하는 **게이트 RMSNorm** 뒤 사영입니다. 기본 헤드 차원은 128(qk)이고 value 쪽은 2배 확장이 가능합니다. 코드에는 논문 식에 없는 query 스케일 $1/\sqrt{d_k}$가 들어 있습니다.

## 3. 청크 병렬화 — 게이트가 WY 표현에 접히는 방식

DeltaNet 편 3절의 핵심은 Householder 곱 $\prod_i (I - \beta_i k_i k_i^{\top})$이 항등 더하기 rank-$r$이라는 WY 표현이었습니다. 게이트가 들어오면 이 곱이 어떻게 되는지가 문제입니다.

답은 "거의 그대로"입니다. $\alpha_i$가 **스칼라**이므로 행렬 곱과 가환이고, 게이트된 전이의 곱은 게이트 없는 곱에 누적 감쇠 $\gamma^r = \prod_{i \le r}\alpha_i$를 곱한 것입니다.

$$
F^r = \prod_{i=1}^{r}\alpha_i(I - \beta_i k_i k_i^{\top}) = \gamma^r P^r = \gamma^r\Big(I - \sum_{i=1}^{r} w_i k_i^{\top}\Big)
$$

즉 $w_i$의 재귀와 UT 변환은 DeltaNet 그대로이고, 청크 끝에서 $\gamma^C$를 한 번 곱하면 됩니다. 쓰기 항은 조금 다릅니다. 토큰 $i$의 쓰기 $u_i k_i^{\top}$는 그 뒤 토큰들의 감쇠 $\gamma^r / \gamma^i$만큼 줄어드므로

$$
G^r = \sum_{i=1}^{r} \frac{\gamma^r}{\gamma^i}\,\tilde{u}_i k_i^{\top}, \qquad \tilde{u}_r = \beta_r\Big(v_r - \sum_{i<r}\tilde{u}_i\,\frac{\gamma^r}{\gamma^i}\,(k_i^{\top}k_r)\Big)
$$

의사 값 $\tilde{u}$의 재귀에서 $k_i^{\top}k_r$이 감쇠 비 $\gamma^r/\gamma^i$로 가중된 것이 유일한 차이입니다. 행렬로 쓰면 DeltaNet 편의 삼각 선형계에서 $KK^{\top}$이 $\Gamma \odot KK^{\top}$으로 바뀝니다($\Gamma_{ij} = \gamma^i/\gamma^j$, $i \ge j$).

$$
\tilde{U} = \big(I + \mathrm{tril}(\mathrm{diag}(\beta)(\Gamma \odot KK^{\top}), -1)\big)^{-1}\mathrm{diag}(\beta)\,V
$$

청크 재귀는 감쇠된 query·key·상태로 씁니다. $\overleftarrow{q}_r = \gamma^r q_r$(청크 처음까지 감쇠), $\overrightarrow{k}_r = (\gamma^C/\gamma^r)k_r$(청크 끝까지 감쇠), $\overrightarrow{S} = \gamma^C S$로 두면

$$
S_{[t+1]} = \overrightarrow{S}_{[t]} + \big(\tilde{U}_{[t]} - \overleftarrow{W}_{[t]}S_{[t]}^{\top}\big)^{\top}\overrightarrow{K}_{[t]}, \qquad
O_{[t]} = \overleftarrow{Q}_{[t]}S_{[t]}^{\top} + \big(Q_{[t]}K_{[t]}^{\top}\odot M\big)\big(\tilde{U}_{[t]} - \overleftarrow{W}_{[t]}S_{[t]}^{\top}\big)
$$

DeltaNet의 청크 식과 나란히 놓으면, 상태에 $\gamma^C$를 곱하고 query·key·$W$에 방향별 감쇠를 붙인 것 외에는 같습니다. 구현은 감쇠를 로그 공간의 누적합으로 다루고 삼각 풀기는 fp32로 합니다. 논문의 표현으로 "원래 delta rule에 비해 미미한 오버헤드"이고, 실측 학습 처리량(1.3B, H100 한 장)은 DeltaNet 45.9K, Gated DeltaNet 46.1K, Mamba-2 48.1K 토큰/초로 **Mamba-2보다 4% 느립니다.** 2K 길이에서 FlashAttention-2를 쓴 Transformer++는 55.0K로 가장 빠르지만 16K에서는 26.5K로 떨어지고 선형 모델들은 그대로입니다.

이 유도에서 게이트가 **스칼라**라는 것이 결정적이었습니다. 감쇠가 채널별 벡터이면 Householder와 가환이 아니고, WY 표현이 그대로 서지 않습니다. 이것이 7절 Kimi Linear의 문제입니다.

## 4. 세 규칙이 각각 어디서 무너지는가 — S-NIAH

논문은 RULER의 단일 바늘 찾기 세 변종으로 세 규칙을 가릅니다. 1.3B 모델, 하이브리드 없음, 트랜스포머 없음. 위 그림 왼쪽이 Table 2 전체입니다.

**S-NIAH-1**은 반복되는 합성 문맥에 패스키 하나를 숨깁니다. 저장할 것이 거의 없으니 **장기 보존**만 시험합니다. DeltaNet은 8K까지 98.8로 완벽하고, Mamba-2는 4K에서 65.4, 8K에서 30.4로 무너집니다. 게이트가 오래된 정보를 너무 빨리 잊습니다. Gated DeltaNet은 91.8로 그 사이입니다. 논문의 첫 관찰 "**감쇠는 보존을 해친다**"입니다.

**S-NIAH-2**는 실제 에세이 안에 숫자를 숨깁니다. 관련될 수 있는 정보를 전부 저장해야 하니 **기억 관리**를 시험합니다. 이번엔 DeltaNet이 무너집니다. 2K에서 45.6, 4K에서 18.6. 비우는 수단이 없어 상태가 포화하고 정보가 겹쳐 구분이 안 됩니다(논문 용어로 memory collision). Mamba-2는 4K에서 56.2, Gated DeltaNet은 92.2입니다. 두 번째 관찰 "**게이트는 거르기를 돕는다**"입니다.

**S-NIAH-3**은 값을 숫자 대신 UUID로 바꿉니다. 복잡한 패턴의 **암기**를 시험합니다. Mamba-2가 1K에서부터 64.4로 뒤지고 4K에서 4.6으로 무너지는 반면 Gated DeltaNet은 2K에서 84.2입니다. 세 번째 관찰 "**delta rule은 암기를 돕는다**"입니다.

세 과제를 함께 보면 그림이 명확합니다. 게이트만 있으면 보존과 암기에서 지고, delta rule만 있으면 관리에서 지며, 둘을 합친 것이 세 곳 모두에서 최선이거나 그에 가깝습니다. 다만 Gated DeltaNet도 S-NIAH-2 8K에서 29.6, S-NIAH-3 4K에서 27.6으로 무너집니다. 고정 크기 상태의 용량 한계를 게이트가 **미루는** 것이지 없애는 것이 아닙니다.

![Gated DeltaNet: S-NIAH and Qwen3-Next memory](/images/gdn.png)

> 왼쪽: 1.3B 순수 모델의 S-NIAH 정확도(Table 2 전체). S-NIAH-1에서는 게이트만 있는 Mamba-2가 8K에서 30.4로 무너지고, S-NIAH-2에서는 delta rule만 있는 DeltaNet이 4K에서 18.6으로 무너지며, S-NIAH-3에서는 Mamba-2가 4.6으로 무너진다. 둘을 합친 Gated DeltaNet은 세 곳 모두 최선이거나 그에 가깝다. 오른쪽: Qwen3-Next의 config에서 계산한 시퀀스당 메모리. GDN 36층의 상태는 36 MiB로 상수이고, 문맥에 비례하는 부분은 attention 12층의 KV(토큰당 2 KiB × 12)뿐이다. 256K에서 6.1 GiB로, 같은 층을 전부 attention으로 채울 때의 4분의 1, Qwen3-32B의 약 10분의 1이다. 직접 계산해 그린 그림이다.

## 5. 실제 언어 모델 — 격차는 얼마인가

1.3B 모델을 FineWeb-Edu 100B 토큰, 길이 4K로 학습한 결과입니다. 여덟 개 상식 과제 평균과 여섯 개 회상 과제(2K 절단) 평균, LongBench 14개 과제 평균을 같이 봅니다.

| 1.3B / 100B | Wiki ppl | LMB ppl | 상식 평균 (8) | 회상 평균 (6) | LongBench (14) |
|---|---|---|---|---|---|
| Transformer++ | 18.53 | 18.32 | 52.25 | 37.0 | 11.0 |
| Mamba | 17.92 | 15.06 | 53.12 | 21.0 | 14.6 |
| Mamba-2 | 16.56 | 12.56 | 54.89 | 29.8 | 13.5 |
| DeltaNet | 17.71 | 16.88 | 52.14 | 26.2 | 13.6 |
| Gated DeltaNet | 16.42 | 12.17 | 55.32 | 30.6 | 16.6 |
| GDN-H1 (+SWA) | 16.07 | 12.12 | 56.40 | 39.0 | 17.8 |
| GDN-H2 (+Mamba-2+SWA) | 15.91 | 12.55 | 56.18 | 40.1 | 18.4 |

정직하게 읽으면 세 가지입니다.

첫째, 제목의 "Mamba-2 개선"은 상식 평균에서 **+0.43점**(55.32 대 54.89)이고 perplexity에서 16.42 대 16.56입니다. 작습니다. DeltaNet 대비로는 +3.18로 큽니다. 즉 이 논문에서 큰 이득은 delta rule에 게이트를 더한 쪽이지, Mamba-2에 delta rule을 더한 쪽이 아닙니다.

둘째, 회상 과제에서 순수 선형 모델은 전부 트랜스포머에 큽니다. Gated DeltaNet 30.6 대 Transformer++ 37.0이고, 세부를 보면 FDA(23.7 대 25.3), TriviaQA, NQ에서는 Mamba-2가 Gated DeltaNet을 앞섭니다. 논문은 이를 "명령 정렬이 안 된 작은 모델의 반복 오류"로 설명합니다.

셋째, 큰 격차는 LongBench(16.6 대 13.5)와 앞 절의 S-NIAH에 있고, 하이브리드는 회상에서도 트랜스포머를 넘습니다(H2 40.1). 2K 창의 sliding window attention을 한 층 걸러 넣는 것만으로 회상 평균이 30.6에서 39.0으로 오릅니다. Hybrid 편과 DeltaNet 편에서 본 결론이 세 번째 반복됩니다.

Ablation(400M/15B)에서는 게이트를 빼면(순수 delta rule) 평균 perplexity가 27.35에서 30.87로 오르고, 짧은 conv를 빼면 28.95, 출력 게이트를 빼면 29.12, L1 정규화로 바꾸면 30.18~30.79입니다. 출력 RMSNorm은 27.55로 거의 차이가 없습니다. 헤드 차원은 64에서 28.31, 128에서 27.35, 256에서 27.13으로 128이 "성능과 효율의 절충"입니다.

## 6. Qwen3-Next — config에서 읽는 채택

2025년 9월 Qwen3-Next-80B-A3B가 이 층을 제품에 넣었습니다. Qwen의 설명은 "체계적 실험에서 Gated DeltaNet이 sliding window attention이나 Mamba-2보다 강한 문맥 내 학습 능력을 보였고, 표준 attention과 3:1로 섞으면 어떤 단일 구조보다 일관되게 나았다"입니다. config.json에서 그 구조를 읽으면

- `full_attention_interval: 4` → 48층 중 $(i+1) \bmod 4 = 0$인 12층이 gated attention, **36층이 Gated DeltaNet**
- GDN: `linear_num_value_heads: 32`, `linear_num_key_heads: 16`, 헤드 차원 128/128, conv 커널 4 → query·key 헤드 16개를 value 헤드 32개에 맞춰 반복하는 GVA
- attention: 헤드 16, KV 헤드 2, 헤드 차원 256, RoPE는 앞 25% 차원에만, 출력 sigmoid 게이트
- MoE: 전문가 512, 활성 10 + 공유 1, 전문가 중간 차원 512. 총 80B, 활성 3B

HF 구현의 핵심 줄은 논문과 flash-linear-attention 그대로입니다. `beta = b.sigmoid()`, `g = -A_log.exp() * softplus(a + dt_bias)`, query·key L2 정규화는 커널 안에서, 출력은 `RMSNormGated` 뒤 `out_proj`. 차이는 $A_{\log}$ 초기화 범위가 $(0.01, 16)$이라는 것과 qkv·게이트 사영이 하나로 합쳐져 있다는 것뿐입니다.

### 6.1 메모리 산수

이 구조가 왜 3:1인지는 메모리로 보입니다. GDN 층의 상태는 시퀀스당 `[32 heads, 128, 128]` = 524,288개 원소, bf16으로 **1 MiB**이고 문맥 길이와 무관합니다(conv 상태 64 KiB 추가). Attention 층의 KV 캐시는 토큰당 $2 \times 2 \times 256 \times 2 = 2$ KiB로 문맥에 비례합니다. 위 그림 오른쪽이 이 산수입니다.

| 문맥 | 12 attention 층 KV | 36 GDN 층 상태 | 합계 | 48층 전부 attention이면 | Qwen3-32B (64층 × 8 KV 헤드) |
|---|---|---|---|---|---|
| 32K | 768 MiB | 36 MiB | 804 MiB | 3 GiB | 8 GiB |
| 256K | 6 GiB | 36 MiB | 6.0 GiB | 24 GiB | 64 GiB |

같은 층 수를 전부 attention으로 채우는 것에 비해 4분의 1, 비교 대상인 dense Qwen3-32B에 비해 약 10분의 1입니다. Qwen이 주장하는 "32K 이상에서 10배 처리량"의 분모가 이 KV 읽기입니다. 이 표는 config에서 제가 계산한 것이고 Qwen의 수치가 아닙니다. Qwen의 공식 수치는 비율뿐입니다. prefill 4K에서 약 7배, 32K 이상에서 10배 이상, decode 4K에서 약 4배, 32K 이상에서 10배 이상, 학습은 Qwen3-32B의 9.3% 연산. 모델 카드는 "효율 개선은 구현에 크게 의존한다"고 덧붙입니다.

RULER 1M(YaRN)에서 4K 98.5, 128K 96.0, 512K 86.9, 1M 80.3입니다. 긴 문맥의 회상은 36개 GDN 층의 36 MiB가 아니라 12개 attention 층이 담당한다는 것이 5절의 결론과 일치합니다.

## 7. Kimi Linear — 게이트를 채널별로

같은 해 10월 Moonshot의 Kimi Linear(48B-A3B)는 Gated DeltaNet의 스칼라 게이트를 **채널별 벡터**로 바꾼 KDA(Kimi Delta Attention)를 씁니다. Kimi의 표기(상태를 $d_k \times d_v$로 전치)로

$$
S_t = (I - \beta_t k_t k_t^{\top})\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^{\top}, \qquad \alpha_t \in [0, 1]^{d_k}
$$

감쇠를 먼저 채널별로 곱하고 Householder를 적용합니다. 감쇠 벡터는 저랭크 사영 뒤 GDN과 같은 $\exp(-\exp(A)\cdot\mathrm{softplus})$을 채널마다 적용한 것이고, 출력 게이트는 SiLU 대신 sigmoid입니다(그들의 ablation에서 SiLU 게이트가 "실질적으로 나쁨": 검증 ppl 5.81 대 5.67).

3절에서 예고한 문제가 여기서 나타납니다. 감쇠가 벡터이면 Householder와 가환이 아니라 WY 표현이 그대로 서지 않고, 일반적인 "대각 더하기 저랭크(DPLR)" 전이는 GLA처럼 로그 공간 계산과 fp32 2차 청킹이 필요해 반정밀도 행렬 곱을 다 못 씁니다. Kimi의 해법은 저랭크 항의 두 벡터를 $a_t = \beta_t k_t$, $b_t = k_t \odot \alpha_t$로 **키에 묶는** 것입니다. 그러면 2차 청크 행렬 계산이 4개에서 2개로 줄고 행렬 곱 3개가 사라져, 커널 시간이 8K 길이에서 DPLR 7.55ms 대 KDA 3.88ms입니다. FLOPs는 $6Td_h^2 + 3TCd_h + TC^2$, $C = 64$.

같은 구조·같은 데이터(1.4T 토큰)로 GDN 하이브리드와 통제 비교한 결과, 짧은 문맥에서는 KDA > GDN-H > MLA 순이지만(MMLU-Pro 51.0 / 47.9 / 47.2), **128K 긴 문맥에서는 GDN-H가 MLA 아래로** 내려갑니다(RULER 80.5 대 81.3, 평균 51.2 대 52.2). KDA(RoPE 없는 MLA와 3:1)는 54.5로 위입니다. 합성 과제에서는 차이가 더 큽니다. 1024 길이 palindrome에서 KDA는 4K 스텝에 94.2%에 도달하는데 GDN은 6K까지 0이다가 16K에서 98.7%입니다. 채널별 게이트가 학습을 빠르게 한다는 것이지, 최종 표현력이 다르다는 근거는 아닙니다.

## 8. 그 뒤

Qwen은 이 층을 표준으로 삼았습니다. Qwen3.5(2026년 2월, 397B-A17B·35B-A3B·27B)는 모두 `full_attention_interval: 4`의 GDN + gated attention이고, Qwen3.8-Flash-Next(2026년 8월, "Qwen4 구조의 조기 미리보기")는 GDN 36층 + 희소 attention 12층에 출력 게이트를 sigmoid로 바꿨습니다. 그 기술 보고서의 ablation(25B-A3B, 480B 토큰)은 9개 벤치마크 평균 전체 attention 49.87, SWA 하이브리드 51.15, **GDN 하이브리드 53.81**입니다.

반대 방향도 있습니다. NVIDIA Nemotron 3는 GDN이 아니라 Mamba-2와 attention의 하이브리드이고, MiniMax-M2는 "어떤 효율적 attention 변종도 제품 환경에서 전체 attention 품질을 안정적으로 맞추지 못했다"며 Lightning Attention을 버리고 전체 attention으로 돌아갔습니다. DeepSeek-V4는 선형 층 없이 압축·희소 attention을 씁니다. 2026년 시점에 선형 층의 채택은 Qwen·Kimi 계열에 집중되어 있습니다.

RWKV-7은 같은 방향을 더 밀어 감쇠도 벡터, 학습률도 벡터, 지우는 키와 쓰는 키를 분리한 $S_t = S_{t-1}(\mathrm{diag}(w_t) - \hat{\kappa}_t^{\top}(a_t \odot \hat{\kappa}_t)) + v_t^{\top}k_t$를 씁니다. Gated DeltaNet 논문 스스로 이것을 "대각 더하기 저랭크 전이를 쓰는 더 완화된 형식의 동시 연구"로 적습니다.

## 9. 논문이 말하지 않는 것

- **규모.** 1.3B/100B가 최대이고 가중치는 공개되지 않았습니다. 본문이 "400M과 1.3B 두 규모"라 하지만 400M의 벤치마크별 표는 없고 ablation뿐입니다.
- **400M 이상의 "Mamba-2 개선" 폭.** 1.3B에서 +0.43점. 제품 규모의 비교는 Qwen·Kimi의 내부 실험뿐입니다.
- **S-NIAH는 합성이고 하이브리드·트랜스포머 행이 없습니다.** S-NIAH-3는 4K까지만입니다.
- **길이 외삽은 "혼합".** GovReport·QMSum에서는 GDN이 최선이지만 PG19 20K에서는 Mamba-2(12.71)가 GDN(13.35)보다 낮습니다. 논문의 그림 데이터에서 Qasper의 DeltaNet과 GDN-H1 곡선이 바이트 단위로 같은데, 어느 쪽이 맞는지 알 수 없습니다.
- **$\beta \in (0, 1)$.** 음의 고유값(DeltaNet 편 6절)은 각주와 코드 옵션(`allow_neg_eigval`, 기본 꺼짐)에만 있고, Qwen3-Next·Kimi 모두 sigmoid를 씁니다. 상태 추적 능력에 대한 주장은 배포 모델에 적용되지 않습니다.
- **정밀도.** 논문은 수치 정밀도를 언급하지 않습니다. 구현은 감쇠를 로그 공간에, 삼각 풀기를 fp32에 두고, Qwen3.5·3.8 config는 상태를 `float32`로 고정합니다. MiniMax-M2 보고서는 "저정밀도 저장에 대한 민감성"을 선형 attention을 채택하지 않은 이유로 듭니다.
- **헤드 차원 불일치.** NVIDIA 학습 코드의 1.3B 기본값은 qk 헤드 차원 200인데 논문 ablation의 선택은 128입니다. 1.3B 실험이 어느 쪽인지 확인할 수 없습니다.
- **Qwen의 처리량.** 비율만 있고 절대 토큰/초가 없으며, 비교 대상은 dense 32B입니다.

## 10. 결론

Gated DeltaNet을 한 줄로 요약하면 "**delta rule의 SGD에 적응적 weight decay를 붙인 것**"입니다. 전이 $\alpha_t(I - \beta_t k_t k_t^{\top})$에서 $\alpha_t$는 상태 전체를 얼마나 남길지, $\beta_t$는 키 방향 하나를 얼마나 덮어쓸지를 정하고, 둘은 독립된 축입니다. S-NIAH 세 과제가 그 독립성의 증거입니다. 게이트만 있으면 보존(S-NIAH-1 8K: 30.4)과 암기(S-NIAH-3 4K: 4.6)에서 지고, delta rule만 있으면 관리(S-NIAH-2 4K: 18.6)에서 집니다.

병렬화는 $\alpha_t$가 스칼라라서 공짜에 가까웠습니다. Householder 곱과 가환이므로 WY 표현에 누적 감쇠 $\gamma$를 곱하기만 하면 되고, 처리량은 Mamba-2의 96%입니다. 감쇠를 벡터로 바꾼 Kimi의 KDA가 저랭크 항을 키에 묶는 대가를 치러야 했던 것이 그 반증입니다.

그리고 실제 언어 모델에서의 격차는 상식 과제 +0.43점이 아니라 긴 문맥과 회상 계열에 있으며, 그것마저 순수 모델로는 트랜스포머에 못 미쳐 3:1 하이브리드가 답이었습니다. Qwen3-Next의 config가 그 답을 숫자로 보여 줍니다. 36개 GDN 층의 상태 36 MiB는 문맥과 무관하고, 256K에서의 6 GiB는 전부 12개 attention 층의 것입니다.

*(참고: Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025, arXiv:2412.06464 v3 · 구현 fla-org/flash-linear-attention `gated_deltanet.py`·`ops/gated_delta_rule/`, NVlabs/GatedDeltaNet · Qwen3-Next 모델 카드·config·HF `modeling_qwen3_next.py`·Qwen 블로그(2025-09-11) · Kimi Linear arXiv:2510.26692 v2 · Qwen3.8-Flash-Next 기술 보고서 · RWKV-7 arXiv:2503.14456. 3절의 WY 확장 유도와 6.1절의 메모리 표는 직접 한 것이며, 3절의 처리량과 4·5절의 수치는 논문의 표와 그림 소스에서 옮긴 것입니다. 9절에 적은 대로 400M 벤치마크 표·정밀도·1.3B 헤드 차원은 공개되지 않았습니다. 그림은 직접 그린 것입니다.)*

#Gated-DeltaNet #delta-rule #gating #Qwen3-Next #Kimi-Linear #linear-attention #아키텍처
