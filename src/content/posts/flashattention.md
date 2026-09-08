---
title: "어텐션은 느린 게 아니라, 메모리를 못 참는다"
subtitle: "FlashAttention — n×n 행렬을 만들지 않고 정확히 어텐션을 계산하는 법"
description: "표준 어텐션의 진짜 병목은 FLOPs가 아니라 HBM I/O. tiling + online softmax + recomputation으로 n×n 행렬을 아예 만들지 않고, 정확히 같은 어텐션을 메모리 O(n)으로 계산한다."
pubDate: 2026-09-02
order: 4
track: "추론 효율"
tags: ["FlashAttention","attention","추론효율","롱컨텍스트"]
hero: "/images/flash_memory.png"
---

지난 편에서 표준 어텐션은 시퀀스 길이에 대해 $O(n^2)$ 메모리를 쓴다고 했습니다. $n\times n$ 어텐션 행렬을 만들어 HBM에 통째로 쓰고 다시 읽기 때문이죠. 128K 토큰이면 헤드 하나당 그 행렬만 **32GB**입니다.

**FlashAttention의 통찰은 한 문장입니다: 그 $n\times n$ 행렬을 아예 만들지 마라.** 근사가 아니라 *정확히 같은* 어텐션을, 행렬을 HBM에 한 번도 올리지 않고 계산합니다. 이 글에서는 그 트릭 — tiling + online softmax + recomputation — 을 유도까지 따라갑니다. 용어는 영어를 그대로 씁니다.

## 1. 진짜 병목은 FLOPs가 아니라 HBM I/O

표준 어텐션은 세 단계입니다.

$$
S = \frac{QK^\top}{\sqrt d} \in \mathbb{R}^{n\times n}, \qquad P=\mathrm{softmax}(S), \qquad O = PV.
$$

여기서 $Q,K,V\in\mathbb{R}^{n\times d}$는 각각 쿼리·키·밸류 행렬입니다(행 하나 = 토큰 하나, $n$=토큰 수, $d$=헤드 차원). $S$의 $(i,j)$ 원소 $S_{ij}=q_i^\top k_j/\sqrt d$는 **쿼리 $i$가 키 $j$에 대해 갖는 점수(score)**, $P$는 $S$의 각 행에 softmax를 취한 **어텐션 가중치**($n\times n$), $O\in\mathbb{R}^{n\times d}$는 최종 출력입니다.

문제는 가운데 두 개, $S$와 $P$($n\times n$)를 **HBM에 썼다가 다시 읽는다**는 것입니다. GPU 메모리 계층을 보면 왜 이게 치명적인지 드러납니다.

- **SRAM (on-chip)**: 수십~수백 KB, 대역폭 ~19 TB/s — 매우 빠르지만 매우 작음
- **HBM (off-chip)**: 수십 GB, 대역폭 ~1.5–3 TB/s — 크지만 느림

어텐션은 연산 대비 메모리 이동이 많은(low arithmetic intensity) **memory-bound** 연산입니다. 즉 GPU는 $n\times n$ 행렬을 HBM에 실어 나르느라 시간을 쓰지, 연산으로 바쁜 게 아닙니다. **그러니 줄여야 할 것은 FLOPs가 아니라 HBM 왕복입니다.**

## 2. 아이디어: 블록으로 쪼개 SRAM 안에서 끝낸다 (tiling)

$Q,K,V$를 블록으로 나눠, 한 번에 한 쌍의 블록만 SRAM에 올리고, **SRAM 안에서** 부분 어텐션을 계산해 출력에 누적합니다. $n\times n$ 행렬은 어디에도 통째로 존재하지 않습니다.

```
        K,V 블록 →  [K1 V1] [K2 V2] [K3 V3] ...
Q 블록
 [Q1] ──►  각 (Qi, Kj) 쌍을 SRAM에 올려 부분 softmax·부분 출력 계산
 [Q2] ──►  → running 통계로 O에 누적 (n×n 은 HBM에 안 씀)
 ...
```

걸림돌이 하나 있습니다. softmax는 **행 전체의 정규화**(분모 $\sum_l e^{s_l}$)가 필요한데, 블록을 하나씩만 보면 전체 합을 모릅니다. 이걸 **online softmax**가 해결합니다.

## 3. Online softmax — 한 항씩 유도

먼저 기호부터 정확히 정해둡시다. 어텐션은 **하나의 쿼리** $q_i$를 고정하고, 그 쿼리가 각 **키** $k_j$에 대해 갖는 **점수(score)**

$$
s_j \;=\; \frac{q_i^\top k_j}{\sqrt d}\qquad (j=1,\dots,n)
$$

를 계산하는 데서 시작합니다. 즉 $s_j$는 실수 하나이고, 위 $S$ 행렬의 $i$행 $j$열 원소 $S_{ij}$와 같은 값입니다($j$는 모든 키를 도는 인덱스). softmax는 이 점수들 **전체**를 정규화하므로, 오버플로를 막으려 최댓값을 빼서 씁니다. 최댓값을 $m=\max_{l} s_l$(모든 키에 대한 점수 중 가장 큰 값, $l$도 키를 도는 인덱스)이라 하면

$$
\mathrm{softmax}(s)_j = \frac{e^{s_j-m}}{\sum_{l} e^{s_l-m}}.
$$

걸림돌은 분모 $\sum_l e^{s_l-m}$가 **키 전체를 다 봐야** 나온다는 점입니다. online softmax는 이걸 **키를 하나(또는 한 블록)씩 흘려보내며** 누적으로 만듭니다. 그러려면 세 개의 running(계속 갱신되는) 상태가 필요합니다.

- $m$ — 지금까지 본 점수들의 **running 최댓값** (초기값 $-\infty$)
- $\ell$ — 지금까지의 **running 분모**, 즉 $\sum e^{s-m}$의 누적값인 스칼라 (초기값 $0$)
- $\mathbf{o}$ — 지금까지의 **running 출력**, 즉 $\sum e^{s-m}\,\mathbf{v}$의 누적인 길이 $d$ 벡터 (초기값 $\mathbf 0$)

이제 새 키 하나의 점수 $s$(위 $s_j$ 중 아직 안 본 것 하나)와 그 키의 **밸류 벡터** $\mathbf{v}\in\mathbb{R}^{d}$를 만나면, 세 상태를 이렇게 갱신합니다.

$$
m' = \max(m,\,s),\qquad
\ell' = \ell\,e^{\,m-m'} + e^{\,s-m'},\qquad
\mathbf{o}' = \mathbf{o}\,e^{\,m-m'} + e^{\,s-m'}\,\mathbf{v}.
$$

여기서 $e^{\,m-m'}$이 결정적입니다. 새 최댓값 $m'$이 이전 $m$보다 커지면, **이전까지 누적한 $\ell,\mathbf{o}$를 새 기준으로 다시 스케일**해 주는 보정항이죠. 모든 블록을 처리한 뒤 최종 출력은 $\mathbf{o}_i = \mathbf{o}/\ell$.

쉽게 말해, **전체 점수를 한 번에 볼 필요 없이** 최댓값·분모·출력을 블록마다 보정하며 누적하면 **정확히 같은** softmax가 나옵니다. 그래서 $n\times n$을 저장할 이유가 사라집니다.

## 4. IO 복잡도 — 왜 빨라지나

SRAM 크기를 $M$(원소 수)이라 하면, FlashAttention의 HBM 접근 횟수는

$$
\Theta\!\left(\frac{n^2 d^2}{M}\right)\quad\text{인 반면, 표준 어텐션은}\quad \Theta(nd + n^2).
$$

보통 $d^2 \ll M$이라 표준보다 훨씬 적습니다. 게다가 저장 메모리는 $n\times n$ 대신 softmax 통계 $(m,\ell)$만 남기므로 **$O(n)$** — 그래서 긴 컨텍스트가 OOM 없이 돕니다.

![FlashAttention은 n×n 어텐션 행렬을 없앤다](/images/flash_memory.png)
> 헤드당 어텐션 스크래치 메모리(양쪽 로그). 표준은 $O(n^2)$으로 128K에서 헤드 하나당 ~32GB까지 치솟아 OOM을 부르지만, FlashAttention은 $O(n)$으로 평탄합니다. 결과는 **정확도 손실 0(exact)**, 메모리 $O(n^2)\to O(n)$, 벽시계 속도 2–4배(주로 HBM 왕복 감소).

## 5. Backward — 저장 대신 재계산(recomputation)

역전파에는 $P$($n\times n$)가 필요합니다. 그런데 이걸 저장하면 도로 $O(n^2)$ 메모리죠. FlashAttention은 저장하는 대신, 앞서 남겨둔 출력 $O$와 통계 $(m,\ell)$로 **backward에서 $P$를 다시 계산**합니다. FLOPs를 조금 더 쓰고 메모리를 아끼는 교환인데, memory-bound인 어텐션에선 이 교환이 이득입니다.

## 6. v2 → v3

- **FlashAttention-2**: non-matmul 연산을 줄이고, 시퀀스 블록과 워프에 걸쳐 병렬화를 개선해 v1 대비 약 2배.
- **FlashAttention-3 (Hopper)**: FP8과 비동기 실행(TMA, warp-specialization, producer/consumer)으로 H100의 성능을 제대로 끌어냅니다.

## 7. 결론

FlashAttention은 어텐션을 **바꾸지 않습니다**(출력이 정확히 동일). 대신 그 **실행 방식**만 바꿉니다 — $n\times n$을 HBM에 올리지 않는 IO-aware 커널로요. 지난 편의 "어텐션은 memory-bound"라는 통찰을, 근사 없이 커널 레벨에서 실현한 셈입니다. **롱컨텍스트가 실용이 된 결정적 한 걸음**이 바로 여기 있습니다.

#FlashAttention #attention #추론효율 #롱컨텍스트 #IO-aware
