---
title: "K와 V를 저장하지 마라, 압축본 하나만 남겨라"
subtitle: "Multi-head Latent Attention — 헤드 128개를 유지하면서 KV cache를 MQA 수준으로 줄이는 저차원 잠재의 수학"
description: "DeepSeek-V2/V3의 MLA는 key·value를 576차원 잠재 벡터 하나로 압축해 캐시한다. 저차원 사영, 상향 사영을 query 쪽으로 흡수하는 트릭, RoPE가 왜 흡수를 깨는지와 decoupled RoPE의 해법, 캐시·연산 산술까지 유도한다."
pubDate: 2026-09-03
order: 10
track: "아키텍처"
tags: ["MLA","DeepSeek","KV cache","attention","아키텍처"]
hero: "/images/mla_cache.png"
---

지난 편의 GQA는 **헤드 수**를 줄여 KV cache를 줄였습니다. 손잡이는 하나, $G$. 그런데 $G=1$(MQA) 아래로는 내려갈 수 없고, 헤드를 줄일수록 각 헤드의 개성이 사라집니다.

DeepSeek-V2(2024)의 Multi-head Latent Attention(MLA)은 축을 바꿉니다. **헤드는 128개를 그대로 두고, 그 128개 헤드의 key·value 전부를 하나의 작은 잠재 벡터로 압축해 저장**합니다. 토큰당 캐시가 MHA의 32,768개 원소에서 **576개**로 — 57분의 1. 그런데도 논문의 ablation에서 품질은 MHA 이상입니다. 이 글은 그게 어떻게 가능한지를 한 줄씩 유도합니다. 특히 왜 RoPE가 문제가 되고 어떻게 우회했는지가 이 설계의 백미입니다. 용어는 영어를 그대로 씁니다.

## 1. 설정과 기호

토큰 $t$의 층 입력을 $h_t\in\mathbb{R}^{d}$라 합니다(DeepSeek-V2: $d=5120$). 헤드 수 $n_h=128$, 헤드 차원 $d_h=128$. 표준 MHA라면 헤드 $i$의 query·key·value는

$$
q_{t,i}=W_i^{Q}h_t,\qquad k_{t,i}=W_i^{K}h_t,\qquad v_{t,i}=W_i^{V}h_t\qquad(\text{각각 }\in\mathbb{R}^{d_h}),
$$

이고 토큰당 캐시는 $2\,n_h d_h = 32{,}768$개 원소(층당)입니다. GQA-8이면 $2\cdot8\cdot128=2{,}048$, MQA면 $256$.

## 2. 저차원 잠재로 압축 — low-rank KV compression

MLA는 key·value를 만들기 전에 **먼저 작은 벡터로 줄입니다.**

$$
c_t^{KV} = W^{DKV}\,h_t\ \in\mathbb{R}^{d_c},\qquad d_c = 512 .
$$

$W^{DKV}\in\mathbb{R}^{d_c\times d}$는 **down-projection**(D), $c_t^{KV}$는 토큰 $t$의 **압축된 KV 잠재(latent)** 입니다. $d_c=512$는 $4d_h$ — 즉 헤드 4개 분량의 차원에 128개 헤드의 key·value 정보를 담습니다. 그다음 이 잠재에서 헤드별 key·value를 **복원**합니다.

$$
k_{t,i}^{C} = W_i^{UK}\,c_t^{KV},\qquad v_{t,i}^{C} = W_i^{UV}\,c_t^{KV},\qquad W_i^{UK},W_i^{UV}\in\mathbb{R}^{d_h\times d_c}.
$$

$W^{UK},W^{UV}$는 **up-projection**(U), 위첨자 $C$는 "content"(잠재에서 복원한 내용 부분)를 뜻합니다. 캐시에는 **$c_t^{KV}$만** 저장합니다. key·value는 필요할 때 다시 만들면 되니까요.

이건 사실 행렬 $W^{K}$(전체 헤드를 세로로 쌓으면 $n_hd_h\times d = 16384\times5120$)를 **랭크 $d_c=512$짜리 두 행렬의 곱** $W^{UK}W^{DKV}$로 제한한 것입니다. 쉽게 말해, 128개 헤드의 key가 서로 독립적인 16,384차원이 아니라 **512차원 부분공간 안에서만 움직인다**고 가정하는 겁니다. GQA가 "헤드 몇 개는 서로 같다"는 강한 제약이었다면, MLA는 "헤드들은 저차원에서 서로 다르다"는 훨씬 부드러운 제약입니다. 같은 이유로 query도 $c_t^{Q}=W^{DQ}h_t\in\mathbb{R}^{d_c'}$($d_c'=1536$)로 압축한 뒤 $q_{t,i}^{C}=W_i^{UQ}c_t^{Q}$로 복원하는데, 이는 캐시가 아니라 **학습 시 활성 메모리**를 줄이기 위한 것입니다.

## 3. 복원하지 않아도 된다 — 흡수(absorption) 트릭

"필요할 때 다시 만든다"면 디코딩마다 $L$개 토큰의 key를 전부 복원해야 하니 연산이 늘어나는 것 아닐까요? 여기가 첫 번째 핵심입니다. 헤드 $i$의 어텐션 점수를 써 봅시다.

$$
q_{t,i}^{\top}k_{j,i}^{C}
= \big(W_i^{UQ}c_t^{Q}\big)^{\top}\big(W_i^{UK}c_j^{KV}\big)
= (c_t^{Q})^{\top}\,\underbrace{(W_i^{UQ})^{\top}W_i^{UK}}_{\text{고정 행렬}}\,c_j^{KV}.
$$

$j$(과거 토큰)에 의존하는 것은 $c_j^{KV}$뿐이고, $W_i^{UK}$는 파라미터라 고정입니다. 그러니 **$W_i^{UK}$를 query 쪽으로 옮겨** 놓으면

$$
\tilde q_{t,i} := (W_i^{UK})^{\top}q_{t,i}\in\mathbb{R}^{d_c},\qquad
q_{t,i}^{\top}k_{j,i}^{C} = \tilde q_{t,i}^{\top}\,c_j^{KV}.
$$

즉 **query를 잠재 공간으로 보내면, 캐시된 $c_j^{KV}$와 직접 내적**하면 됩니다. key를 복원할 필요가 없습니다. value도 같습니다.

$$
o_{t,i}=\sum_j p_{t,ij}\,v_{j,i}^{C}=\sum_j p_{t,ij}\,W_i^{UV}c_j^{KV}=W_i^{UV}\Big(\sum_j p_{t,ij}\,c_j^{KV}\Big),
$$

$p_{t,ij}$는 softmax 후 어텐션 가중치입니다. **잠재에 대해 가중합을 먼저 하고 마지막에 한 번만 up-projection**하면 됩니다(또는 $W_i^{UV}$를 출력 사영 $W^{O}$에 미리 곱해 둡니다).

이렇게 보면 MLA의 디코딩은 **"512차원짜리 key·value 하나를 128개 헤드가 공유하는 MQA"** 와 같습니다. 단 각 헤드가 그 공유 벡터를 **자기만의 사영 $\tilde q_{t,i}$로 다르게 읽는다**는 점이 MQA와 다릅니다. GQA/MQA에서 헤드의 개성이 사라졌던 이유가 key·value의 사영이 같아서였다면, MLA는 저장은 공유하되 **읽는 방식은 헤드마다 다르게** 남겨 둔 것입니다.

## 4. RoPE가 흡수를 깬다 — 그리고 decoupled RoPE

두 번째 핵심. 현대 LLM은 위치 정보를 **RoPE**로 넣습니다(롱컨텍스트 편에서 자세히). RoPE는 query와 key에 위치 $t$에 의존하는 **회전 행렬** $R_t$를 곱합니다. 회전이라 $R_t^{\top}R_j=R_{j-t}$가 성립하고, 그 덕에 점수가 상대 위치 $j-t$에만 의존합니다.

$$
(R_tq_t)^{\top}(R_jk_j) = q_t^{\top}R_t^{\top}R_j\,k_j = q_t^{\top}R_{j-t}\,k_j .
$$

이제 MLA의 key에 RoPE를 적용하면 $k_{j,i}=R_j\,W_i^{UK}c_j^{KV}$이고, 점수는

$$
q_{t,i}^{\top}R_t^{\top}R_j\,W_i^{UK}\,c_j^{KV}.
$$

§3에서는 $q$와 $c_j$ 사이에 **고정 행렬** $W_i^{UK}$만 있어서 query 쪽으로 옮길 수 있었습니다. 그런데 이제 그 사이에 **$j$에 의존하는 $R_j$** 가 끼어 있습니다. $R_jW_i^{UK}$는 과거 토큰마다 다른 행렬이라 미리 곱해 둘 수 없고, 결국 디코딩마다 모든 $j$에 대해 key를 복원해야 합니다. 흡수 트릭이 무너집니다. 쉽게 말해, **위치 회전이 압축과 복원 사이에 끼어들어 순서를 바꿀 수 없게** 만듭니다.

DeepSeek의 해법은 **위치 정보를 별도의 작은 채널로 분리**하는 것입니다(decoupled RoPE).

$$
k_t^{R} = \mathrm{RoPE}\big(W^{KR}h_t\big)\in\mathbb{R}^{d_h^{R}},\qquad
q_{t,i}^{R} = \mathrm{RoPE}\big(W_i^{QR}c_t^{Q}\big)\in\mathbb{R}^{d_h^{R}},\qquad d_h^{R}=64 .
$$

$k_t^{R}$는 **RoPE가 적용된 작은 key**로, 모든 헤드가 공유합니다(MQA처럼 — 그래서 캐시는 하나만). $q_{t,i}^{R}$는 헤드별 **RoPE 적용 query**입니다. 최종 query·key는 content 부분과 RoPE 부분의 **이어붙이기**이고, 점수는 두 내적의 합입니다.

$$
q_{t,i}=\big[q_{t,i}^{C};\,q_{t,i}^{R}\big],\quad k_{j,i}=\big[k_{j,i}^{C};\,k_j^{R}\big],\qquad
\text{score}_{t,ij}=\frac{\tilde q_{t,i}^{\top}c_j^{KV}+(q_{t,i}^{R})^{\top}k_j^{R}}{\sqrt{d_h+d_h^{R}}} .
$$

앞 항은 §3의 흡수 트릭이 그대로 통하고(RoPE 없음), 뒤 항은 RoPE가 있지만 **$k_j^{R}$를 직접 캐시**하니 복원이 필요 없습니다. 분모의 $\sqrt{d_h+d_h^R}$는 두 부분을 합친 차원에 맞춘 scaling입니다. 결국 캐시는

$$
\underbrace{d_c}_{c_t^{KV}}+\underbrace{d_h^{R}}_{k_t^{R}} = 512+64 = 576 \ \text{원소/토큰/층}.
$$

![MLA 구조와 캐시 비교](/images/mla_cache.png)
> 왼쪽: 입력 $h_t$에서 압축 잠재 $c_t^{KV}$(512)와 RoPE key $k_t^{R}$(64)를 만들어 **이 둘만 캐시**합니다. 헤드별 $k^{C},v^{C}$는 up-projection으로 복원하거나, 아예 복원하지 않고 $W^{UK}$를 query 쪽에 흡수합니다. 오른쪽: 토큰·층당 캐시 원소 수($d_h{=}128$ 기준). MHA 32,768 → GQA-8 2,048 → MLA 576. MLA는 **MQA(256)의 2.25배**에 불과하면서 헤드 128개를 온전히 유지합니다.

## 5. 산술 — 무엇을 얻고 무엇을 내주나

**캐시.** GQA-8 대비 $2048/576\approx3.6$배, MHA 대비 57배 감소. DeepSeek-V2 논문은 자사 dense 67B 대비 KV cache 93.3% 절감, 생성 처리량 5.76배를 보고합니다(MoE 효과 포함).

**디코딩 연산.** 흡수 형태에서 헤드 하나가 key 하나와 내적하는 차원은 $d_c+d_h^{R}=576$ — MHA의 $d_h=128$보다 4.5배 깁니다. 즉 **FLOPs는 늘어납니다.** 하지만 디코딩은 memory-bound(GQA 편 §2)라, 읽기가 57배 줄어든 대가로 연산이 4.5배 느는 교환은 압도적으로 이득입니다. arithmetic intensity로 쓰면, 캐시 원소 하나당 128개 헤드가 각각 2 FLOPs를 쓰니 $2n_h=256$ FLOPs/element — MHA의 2, GQA-8의 16과 비교됩니다.

**prefill.** prefill은 compute-bound이니 흡수 형태(연산 4.5배)가 손해입니다. 그래서 실전 커널(FlashMLA, 2025)은 **prefill에서는 key·value를 복원해 표준 어텐션으로, decode에서는 흡수 형태로** 두 경로를 따로 둡니다.

**품질.** DeepSeek-V2 논문의 ablation(7B급)에서 MLA는 MHA보다 약간 나은 점수를 보였고, GQA·MQA보다는 뚜렷이 좋았습니다. "저차원 부분공간" 가정이 실제 학습된 key·value의 구조와 잘 맞는다는 뜻으로 읽을 수 있습니다 — 다만 이는 실험적 관찰이지 정리는 아닙니다.

| | 캐시 (원소/토큰/층) | 헤드 개성 | 디코딩 FLOPs/헤드/key |
|---|---|---|---|
| MHA | $2n_hd_h=32{,}768$ | 완전 | $2d_h=256$ |
| GQA-8 | $2\cdot8\cdot d_h=2{,}048$ | 그룹 공유 | $256$ |
| MQA | $2d_h=256$ | 없음 | $256$ |
| MLA | $d_c+d_h^R=576$ | 헤드별 $\tilde q_i$ | $2(d_c+d_h^R)=1{,}152$ |

## 6. 그 후 — 변환과 후속

- **기존 모델을 MLA로.** TransMLA(2025)와 MHA2MLA(2025)는 GQA/MHA 체크포인트의 $W^{K},W^{V}$를 저랭크 분해(SVD)해 MLA로 변환한 뒤 소량 재학습하는 방법입니다. GQA 편의 uptraining과 같은 발상이고, 실제로 "GQA는 MLA의 특수한 경우"임을 보일 수 있습니다(그룹 공유를 저랭크 사영으로 표현).
- **DeepSeek-V3.** $d=7168$로 커졌지만 $n_h=128,\ d_h=128,\ d_c=512,\ d_h^{R}=64$는 그대로입니다. MLA + MoE + MTP가 V3의 세 기둥입니다.
- **다음 축.** MLA가 캐시의 **차원**을 줄였다면, 2025년 말 DeepSeek-V3.2의 sparse attention(DSA)처럼 **읽는 토큰 수**를 줄이는 방향이 그 다음입니다 — 롱컨텍스트 트랙의 KV 압축 편에서 다룹니다.

## 7. 결론

MLA의 발상은 셋으로 요약됩니다. **(1)** key·value를 저차원 잠재로 압축해 그것만 저장한다. **(2)** up-projection을 query 쪽으로 흡수해 복원 없이 잠재와 직접 내적한다. **(3)** 흡수를 깨는 RoPE는 64차원짜리 별도 채널로 떼어낸다. 결과는 헤드 128개의 표현력을 유지한 채 캐시를 MQA 수준으로 줄이는 것. GQA가 "헤드 몇 개는 같다"고 뭉뚱그렸다면, MLA는 "헤드들은 저차원에서 다르다"고 정확히 말한 셈입니다.

*(참고: DeepSeek-V2 — arXiv:2405.04434 · DeepSeek-V3 — arXiv:2412.19437 · TransMLA — arXiv:2502.07864 · MHA2MLA — arXiv:2502.14837 · FlashMLA — github.com/deepseek-ai/FlashMLA. 차원 수치는 V2/V3 기술보고서 기준이며, V3.2 관련 언급은 2025년 말 공개로 원문 재확인을 권합니다. 그림은 직접 그린 것입니다.)*

#MLA #DeepSeek #KVcache #attention #아키텍처
