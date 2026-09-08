---
title: "왜 첫 토큰은 굼뜨고 그다음은 술술 나올까"
subtitle: "LLM 추론의 prefill과 decode"
description: "GEMM vs GEMV, arithmetic intensity, KV cache로 보는 LLM 추론의 두 국면 — 왜 prefill은 compute-bound고 decode는 memory-bound인가."
pubDate: 2026-09-02
order: 2
track: "추론 효율"
tags: ["prefill","decode","KV-cache","vLLM","추론"]
hero: "/images/roofline.png"
---

ChatGPT에 긴 문서를 붙여넣어 본 적 있으신가요? **첫 글자가 나오기까지는 몇 초 뜸을 들이다가, 일단 시작되면 토큰이 규칙적으로 뚝뚝 흘러나옵니다.** 같은 모델인데 왜 앞부분만 유독 굼뜰까요?

답은, LLM 추론(inference)이 성격이 완전히 다른 **두 단계 — prefill과 decode** 로 나뉘기 때문입니다. 이 둘은 GPU를 쓰는 방식도, 병목도, 최적화 방법도 정반대입니다. 이 글에서는 두 단계를 정의하고, KV cache·arithmetic intensity 같은 핵심 개념을 수식과 함께 한 단계씩 풀어, 왜 오늘날 LLM 서빙 스택이 지금의 모습이 됐는지까지 짚겠습니다. 용어는 자주 쓰이는 영어 표기를 그대로 씁니다.

## 1. 출발점: LLM은 한 번에 한 토큰씩 뱉는다

트랜스포머 기반 LLM은 **autoregressive**, 즉 다음 토큰을 이전까지의 모든 토큰에 조건부로 하나씩 생성합니다.

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}) .
$$

문제는 "이전까지의 모든 토큰"입니다. $t$번째 토큰을 만들 때 self-attention은 앞선 모든 위치의 Key·Value가 필요합니다. 이걸 매번 처음부터 다시 계산하면 낭비가 크죠. 그래서 추론은 자연스럽게 **두 국면**으로 갈립니다: 프롬프트를 한꺼번에 처리하는 **prefill**과, 그 뒤 토큰을 한 개씩 잇는 **decode**.

## 2. prefill과 decode — 정의

- **Prefill(=prompt encoding)**: 입력 프롬프트의 $n$개 토큰을 **단 한 번의 forward pass로, 병렬로** 통과시킵니다. 이 과정에서 모든 프롬프트 위치의 Key·Value를 계산해 **KV cache에 저장**하고, **첫 번째 출력 토큰**을 만듭니다.
- **Decode(=generation)**: 이후 출력 토큰을 **한 스텝에 하나씩** 생성합니다. 각 스텝은 **새 토큰 1개만** Query로 넣고, self-attention은 캐시에 쌓인 과거 Key·Value 전체를 참조합니다. 새로 계산한 K·V는 다시 캐시에 append.

```
시간 →

[ prefill ]                         [ ---------- decode ---------- ]
프롬프트 n개 토큰                     t1   t2   t3   ...   (한 스텝 = 토큰 1개)
──────────────►  ● 첫 토큰            ●───►●───►●───► ...
1회 forward (병렬)                    매 스텝 forward, KV cache 읽고+append
KV cache 구축                        KV cache 계속 성장
```

쉽게 말해, **prefill은 "읽고 요약하는" 한 번의 큰 계산, decode는 "이어 쓰는" 수백 번의 작은 계산**입니다. 사용자가 체감하는 "첫 글자까지의 침묵"이 prefill, "술술 나오는 스트리밍"이 decode죠.

## 3. 한 forward 안에서 실제로 벌어지는 일 — GEMM vs GEMV

두 단계가 "왜" 그렇게 다른지는, forward pass 안에서 실제로 도는 행렬 연산의 **모양**을 보면 분명해집니다.

트랜스포머 레이어에서 무거운 연산은 대부분 **입력을 가중치 행렬과 곱하는 것**입니다 — QKV projection($W_{qkv}$), output projection($W_o$), 그리고 FFN의 두 행렬($W_1:d\times d_{ff}$, $W_2:d_{ff}\times d$, 보통 $d_{ff}\approx 4d$). 이 가중치들이 파라미터의 대부분이고, 매 forward마다 HBM에서 읽혀야 합니다.

**Prefill — 행렬 × 행렬 (GEMM).** 프롬프트 $n$개 토큰이 한꺼번에 들어오니 입력은 $X\in\mathbb{R}^{n\times d}$. 가중치 곱은

$$
X\,W \;:\; (n\times d)\cdot(d\times d)\;\to\;(n\times d),
$$

즉 **matrix-matrix multiply** (GEMM)입니다. 가중치 $W$를 **한 번 읽어 $n$개 행 전부**에 씁니다. attention도 $QK^\top:(n\times d)(d\times n)\to(n\times n)$의 큰 행렬 연산이라, 연산 유닛이 꽉 찹니다.

**Decode — 행렬 × 벡터 (GEMV).** 한 스텝엔 새 토큰 하나뿐이라 입력은 $x\in\mathbb{R}^{1\times d}$. 같은 곱이

$$
x\,W \;:\; (1\times d)\cdot(d\times d)\;\to\;(1\times d),
$$

즉 **matrix-vector multiply** (GEMV)로 쪼그라듭니다. 가중치 $W$를 읽어 **딱 한 행**에만 씁니다. GEMM에선 한 번 읽은 가중치를 $n$번 재사용했는데, GEMV에선 한 번 읽어 한 번 쓰고 버립니다. **이 재사용 여부가 두 단계의 성격을 가릅니다** — GEMM은 compute-bound, GEMV는 memory-bound.

attention도 모양이 다릅니다. decode 스텝 $t$에서는 쿼리 하나가 캐시된 전체 Key와 만납니다:

$$
q_t\,K_{\le t}^\top \;:\; (1\times d)(d\times t)\;\to\;(1\times t),
$$

그다음 $V_{\le t}$와 가중합. 연산량은 $O(td)$로 작지만, **캐시 전체($t$개의 K·V)를 매 스텝 읽어야** 합니다. 그래서 decode는 스텝마다 두 곳에서 메모리를 긁습니다 — (a) **모델 가중치 전체**($\sim\!2N$ 바이트), (b) **KV cache 전체**($t$에 비례). 긴 컨텍스트에선 (b)가 (a)를 압도하기도 합니다.

**배치가 왜 decode를 살리나.** 서로 다른 $B$개 요청의 "이번 토큰"을 쌓으면 입력이 다시 $(B\times d)$가 되어 $XW$가 GEMV에서 **GEMM으로 복귀**합니다. 가중치를 한 번 읽어 $B$개 토큰에 재사용 → intensity가 $\approx B$로 올라가죠. 단, KV cache는 요청마다 따로라 (b)의 메모리 읽기는 $B$에 비례해 함께 커집니다 — 배치를 키우면 병목이 가중치에서 **KV cache로 옮겨갑니다.**

### 숫자로 보기 (Llama-70B, fp16, A100급 가정)

- 가중치 ≈ **140GB**, HBM 대역폭 ≈ 2TB/s.
- **Decode(배치 1)**: 토큰 하나에 가중치 140GB를 한 번 읽음 → 최소 $140/2000\approx$ **70 ms/token** (연산이 아니라 대역폭이 하한). ITL ≈ 70ms.
- **Prefill(프롬프트 2,000 토큰)**: $2Nn = 2\cdot70\text{e}9\cdot2000 \approx 2.8\times10^{14}$ FLOP. 300 TFLOP/s면 ≈ **0.9초** (연산이 하한). TTFT ≈ 1초.
- 500 토큰 생성이면 decode만 $500\times70\text{ms}\approx$ **35초** — 짧은 프롬프트·긴 출력에선 **전체 시간의 대부분이 decode**입니다.

쉽게 말해, **prefill은 GPU를 꽉 채운 짧고 굵은 한 방, decode는 GPU를 놀리며 대역폭만 축내는 길고 가는 수백 방**입니다.

## 4. KV cache — decode를 가능케 하는 메모리

decode의 각 스텝에서 과거 Key·Value를 다시 계산하지 않으려면 어딘가 저장해야 합니다. 그게 **KV cache**입니다. 크기를 따져 봅시다. 토큰 하나가 차지하는 바이트는

$$
\text{bytes/token} = \underbrace{2}_{K,V} \cdot \, L \cdot d_{\text{kv}} \cdot b_{\text{dtype}} ,
\qquad d_{\text{kv}} = n_{\text{kv-head}} \cdot d_{\text{head}} .
$$

여기서 앞의 **2**는 Key와 Value **두 개**를 저장한다는 뜻이고, $L$은 레이어 수, $b_{\text{dtype}}$은 자료형 한 원소의 바이트 수(fp16이면 2)입니다. $d_{\text{kv}} = n_{\text{kv-head}}\cdot d_{\text{head}}$는 한 레이어에서 K(또는 V)가 차지하는 차원의 총합으로, $n_{\text{kv-head}}$는 **KV 헤드 개수**, $d_{\text{head}}$는 **헤드 하나의 차원**입니다. 여기에 시퀀스 길이 $s$(토큰 수)와 배치 $B$(동시에 처리하는 시퀀스 수)를 곱하면 총량은

$$
\text{KV mem} = 2\,L\,d_{\text{kv}}\,b_{\text{dtype}}\; \cdot s \cdot B .
$$

핵심은 **$s$와 $B$에 정비례**한다는 것입니다. 예로 Llama-2-7B(fp16, $L{=}32$, $d_{\text{kv}}{=}4096$)는 토큰당 $2\cdot32\cdot4096\cdot2 = 0.5\,\text{MB}$. 컨텍스트 32K면 한 시퀀스만 16GB, 128K면 64GB — 모델 가중치(14GB)를 넘어섭니다.

![KV cache는 시퀀스 길이에 정비례해 커진다](/images/kv_cache.png)
> 모델별 KV cache 크기(양쪽 로그). 길이에 선형으로 커지고, MHA를 그대로 쓰면 70B는 128K에서 수백 GB에 달합니다. GQA(KV 헤드를 8개로 공유)가 이를 한 자릿수로 눌러줍니다.

그래서 나온 게 **MQA / GQA**입니다. 여러 attention 헤드가 Key·Value를 **공유**해 $n_{\text{kv-head}}$를 (예: 8로) 줄이면, 위 식의 $d_{\text{kv}}$가 그만큼 작아져 캐시가 크게 절약됩니다. 요즘 대형 모델이 GQA를 기본으로 쓰는 이유죠.

## 5. 왜 성격이 정반대인가 — arithmetic intensity

두 단계의 진짜 차이는 **"연산이 병목이냐, 메모리 대역폭이 병목이냐"**, 이 물음입니다. 이를 재는 잣대가 **arithmetic intensity**(연산 강도) — "메모리에서 읽은 1바이트당 몇 번의 연산을 하는가"입니다.

$$
\text{intensity} = \frac{\text{FLOPs}}{\text{bytes moved}} \;\;[\text{FLOP/byte}] .
$$

파라미터 $N$개 모델의 forward는 토큰당 약 $2N$ FLOP이 들고, 그 연산을 하려면 가중치 $N$개를 HBM에서 읽어야 하니 약 $2N$ 바이트(fp16)를 옮깁니다.

- **Decode(배치 1)**: 토큰 1개를 위해 가중치 전체를 한 번 읽습니다. $\text{intensity} \approx \dfrac{2N}{2N} = 1\ \text{FLOP/byte}$. 극도로 낮죠.
- **Prefill(길이 $n$)**: 같은 가중치를 **한 번 읽어 $n$개 토큰에 재사용**합니다. $\text{intensity} \approx \dfrac{2Nn}{2N} = n$. 프롬프트가 길수록 커집니다.

이 숫자를 GPU의 **ridge point**(peak FLOP/s ÷ 메모리 대역폭)와 비교합니다. A100급이면 대략 $312\,\text{TFLOP/s} \div 2\,\text{TB/s} \approx 156\ \text{FLOP/byte}$.

![Roofline: prefill은 연산에, decode는 대역폭에 묶인다](/images/roofline.png)
> intensity가 ridge point보다 낮으면 **memory-bound**(빗변, 대역폭이 상한), 높으면 **compute-bound**(천장, 연산이 상한). decode는 intensity ≈ 1로 왼쪽 깊숙이(대역폭 병목), prefill은 긴 프롬프트로 오른쪽 천장에 붙습니다.

**즉 decode가 느리고 비싼 근본 이유**는 계산량이 많아서가 아니라, **토큰 하나를 뽑을 때마다 수십~수백 GB의 가중치 전체를 메모리에서 읽어야 하기 때문**입니다. 계산 유닛은 대부분 놀고, HBM 대역폭이 속도를 정합니다. 반대로 prefill은 토큰이 많아 연산 유닛을 꽉 채우니 GPU가 제값을 합니다.

다시 말해, **prefill은 "계산이 아까운" 국면, decode는 "메모리가 아까운" 국면**입니다.

## 6. 그래서 지표도 둘로 나뉜다

추론 성능을 하나의 숫자로 못 재는 이유가 여기 있습니다.

- **TTFT (Time To First Token)**: 첫 토큰까지 걸린 시간 ≈ **prefill 시간**. 프롬프트 길이에 민감.
- **TPOT / ITL (Time Per Output Token / Inter-Token Latency)**: 토큰 사이 간격 ≈ **decode 스텝 시간**. KV cache·가중치 대역폭에 민감.
- **전체 지연**: $\text{latency} \approx \text{TTFT} + (N_{\text{out}}-1)\cdot \text{ITL}$.
- **Throughput**: 초당 토큰 수. 여러 요청을 묶어(batching) 올리며, 보통 **latency와 트레이드오프** 관계.

## 7. 두 병목을 공략하는 최적화들

거의 모든 LLM 서빙 기법은 "prefill의 compute-bound"와 "decode의 memory-bound"라는 두 성질에서 파생됩니다.

- **Continuous batching** (Orca·vLLM): 여러 요청의 decode 스텝을 **한 배치로 묶어** 가중치를 한 번 읽고 여러 토큰에 재사용 → decode의 intensity를 $\approx B$로 끌어올려 대역폭 낭비를 줄입니다. 요청이 끝나면 그 자리에 새 요청을 즉시 채웁니다.
- **PagedAttention** (vLLM): KV cache를 고정 크기 **page**로 관리해 단편화를 없애고 메모리 활용률을 높입니다(가상 메모리의 페이징과 같은 발상).
- **Chunked prefill**: 한 배치에 prefill과 decode가 섞이면, 연산이 무거운 prefill이 GPU를 오래 잡아 **진행 중인 decode들이 멈칫**합니다(ITL 튐). 그래서 긴 prefill을 여러 조각으로 잘라 decode 스텝들과 **번갈아** 실행 — TTFT와 토큰 간 지연의 균형을 잡습니다.
- **Prefill–decode disaggregation** (Splitwise·DistServe): 성질이 다른 두 단계를 **서로 다른 GPU 풀**에 분리 배치해 각자 최적 자원으로 돌립니다.
- **Speculative decoding**: 작은 draft 모델이 여러 토큰을 미리 제안하고, 큰 모델이 **한 번의 forward로 병렬 검증**합니다. memory-bound한 decode에서 "가중치를 한 번 읽어 여러 토큰을 확정"하는 전략입니다. (draft·verify와 rejection sampling의 원리는 **다음 편**에서 따로 다룹니다.)
- **MQA / GQA · KV/weight quantization**: KV cache와 가중치의 **바이트 수 자체**를 줄여, decode가 옮길 메모리를 직접 깎습니다.

## 8. 결론 — 추론은 이제 시스템 문제다

prefill과 decode는 그저 구현 디테일이 아니라, **LLM 서빙의 지도를 그리는 좌표축**입니다. 한쪽은 연산이, 다른 한쪽은 대역폭이 상한을 정하고, TTFT와 ITL이라는 서로 다른 지표를 낳으며, batching부터 speculative decoding까지 모든 최적화의 출발점이 됩니다.

모델을 "얼마나 크게 학습하느냐"만큼이나 "**어떻게 값싸게 추론하느냐**"가 경쟁력이 된 지금, 이 두 단계의 상반된 병목을 이해하는 것은 곧 LLM을 실제로 굴리는 일의 문법을 아는 것입니다.

*다음 편에서는 이 memory-bound 벽을 정면으로 뚫는 **speculative decoding**을 다룹니다 — 작은 모델이 던지고 큰 모델이 받는 draft·verify 구조가, rejection sampling으로 어떻게 출력 분포를 그대로 지키면서 속도를 얻는지를요.*

#LLM추론 #prefill #decode #KVcache #vLLM #트랜스포머
