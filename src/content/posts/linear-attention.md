---
title: "softmax를 걷어내면 트랜스포머는 RNN이 된다"
subtitle: "리니어 어텐션의 수학"
description: "결합법칙 한 줄이 어텐션의 O(n²)을 O(n)으로 바꾸고, 어텐션이 고정 크기 상태를 든 선형 재귀가 되는 과정을 유도까지 따라간다."
pubDate: 2026-09-01
order: 1
track: "아키텍처"
tags: ["linear-attention","transformer","선형재귀","Mamba"]
hero: "/images/complexity_curve.png"
---

긴 컨텍스트가 비싼 이유는 하나로 요약됩니다. **어텐션이 시퀀스 길이에 대해 $O(n^2)$이기 때문**입니다. 128K 토큰을 넣는 순간 어텐션 행렬만 $128\text{K}\times128\text{K}\approx 1.6\times10^{10}$개의 엔트리가 되죠. 리니어 어텐션은 이 제곱을 선형으로 낮추는 한 줄의 대수적 트릭에서 출발합니다.

이 글에서는 그 트릭을 **한 단계씩, 항 하나하나 뜯어가며** 따라갑니다. 수식을 자주 다루지 않는 분도 따라올 수 있게 각 기호의 뜻과 차원(shape)을 매번 짚겠습니다. 왜 그것이 결국 "어텐션 = 선형 RNN"이라는 등가로 이어지는지, 그리고 어디서 정확도를 잃는지까지요.

## 1. 표준 어텐션의 형식화와 복잡도

Query, Key, Value를 $Q,K,V \in \mathbb{R}^{n\times d}$라 합시다. 여기서 $n$은 토큰(단어) 개수, $d$는 각 토큰을 나타내는 벡터의 길이입니다. 세 행렬 모두 "행 하나 = 토큰 하나"입니다. 즉 $q_i, k_j, v_j$는 각각 $i$번째 Query, $j$번째 Key, $j$번째 Value를 나타내는 길이 $d$의 벡터죠.

scaled dot-product attention은

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

한 줄이지만 안에서 벌어지는 일을 풀면 이렇습니다.

- $QK^\top$: $(n\times d)$와 $(d\times n)$의 곱이라 결과는 $n\times n$ 행렬입니다. 이 행렬의 $(i,j)$ 원소는 $q_i^\top k_j$, **Query $i$와 Key $j$가 얼마나 닮았는지**(dot product)를 잰 값입니다.
- $\sqrt d$로 나누는 건 내적 값이 $d$가 클수록 커져 softmax가 뾰족해지는 걸 막는 정규화입니다.
- $\mathrm{softmax}$는 **각 행(row)마다** 적용됩니다. $i$번째 행을 확률처럼 만들어, Query $i$가 각 Key에 얼마나 "주의"를 줄지 가중치로 바꿉니다.

행 단위로 다시 쓰면 $i$번째 출력은

$$
o_i \;=\; \sum_{j=1}^{n} \underbrace{\frac{\exp\!\big(q_i^\top k_j/\sqrt d\big)}{\sum_{l=1}^{n}\exp\!\big(q_i^\top k_l/\sqrt d\big)}}_{\text{가중치 }a_{ij}}\, v_j .
$$

쉽게 말해, **출력 $o_i$는 Value들의 가중 평균**이고, 가중치 $a_{ij}$는 "Query $i$가 Key $j$에 얼마나 주목하는가"입니다.

이제 비용을 봅시다. 병목은 가운데의 $A = \mathrm{softmax}(QK^\top/\sqrt d)\in\mathbb{R}^{n\times n}$입니다.

- **연산량**: $A$는 원소가 $n^2$개, 각 원소가 길이 $d$짜리 dot product이니 $QK^\top$에 $O(n^2 d)$. 뒤의 $AV$도 $(n\times n)(n\times d)$라 $O(n^2 d)$ → 합쳐서 **$O(n^2 d)$**.
- **메모리**: $n\times n$ 행렬 $A$를 통째로 들고 있어야 하므로 **$O(n^2)$**.

$n$이 커지면 $d$는 상수처럼 취급되고, $n^2$이 전부를 지배합니다. 토큰이 2배면 일은 4배가 되는 셈이죠.

## 2. 핵심 관찰: softmax는 "분해가 안 되는" 유사도다

왜 꼭 $n\times n$ 표를 만들어야 할까요? 범인은 $\exp$입니다. 가중치의 분자 $\exp(q_i^\top k_j)$는 Query $i$와 Key $j$를 **한 덩어리로 엉켜서** 담습니다. 만약 이게 $f(q_i)\cdot g(k_j)$처럼 "Query만의 항 × Key만의 항"으로 **쪼개진다면**, Key 쪽을 미리 다 더해두고 Query는 그 요약만 보면 될 텐데, $\exp(q_i^\top k_j)$는 그렇게 깔끔히 쪼개지지 않습니다.

리니어 어텐션의 전제는 딱 여기입니다. 이 "엉킨 유사도"를 **쪼개지는 유사도**로 바꾸자. 즉 어떤 feature map $\phi:\mathbb{R}^d\to\mathbb{R}^m_{\ge 0}$을 도입합니다. 이건 **$d$차원 벡터를 음수 없는 $m$차원 특성 벡터로 보내는 함수**이고, $m$은 그 특성 벡터의 차원(방법마다 다르게 고르는 하이퍼파라미터)입니다. 그러면

$$
\exp\!\big(q_i^\top k_j/\sqrt d\big)\;\longrightarrow\;\phi(q_i)^\top \phi(k_j)
$$

로 근사합니다. $\phi(q_i)^\top \phi(k_j)$는 이미 "Query 쪽 벡터 $\phi(q_i)$"와 "Key 쪽 벡터 $\phi(k_j)$"의 내적이라, **Query 부분과 Key 부분이 분리**돼 있습니다. 이 분리가 3장의 트릭을 가능하게 합니다.

한 가지 조건: $\phi\ge 0$(출력이 음수가 아님)을 요구합니다. 어텐션 가중치가 음수가 되면 분모(정규화항)가 0에 가까워지거나 부호가 뒤집혀, "가중 평균"이라는 확률적 해석이 깨지기 때문입니다.

## 3. 결합법칙 트릭 — 한 항씩 유도

커널을 대입한 출력에서 출발합니다.

$$
o_i \;=\; \frac{\sum_{j=1}^{n}\big(\phi(q_i)^\top\phi(k_j)\big)\,v_j}{\sum_{j=1}^{n}\phi(q_i)^\top\phi(k_j)} .
$$

여기서 핵심은 분자입니다. 한 항씩 뜯어봅시다.

- $\phi(q_i)^\top\phi(k_j)$는 길이 $m$ 벡터 두 개의 내적이니 **스칼라**(숫자 하나)입니다. 이걸 $s_{ij}$라 부릅시다.
- 그럼 분자는 $\sum_j s_{ij}\, v_j$, 즉 스칼라 $s_{ij}$로 Value $v_j$를 가중합한 것.

이제 결정적 관찰: **$\phi(q_i)$는 $j$에 전혀 의존하지 않습니다.** 합은 $j$에 대해 도는데, $\phi(q_i)^\top$는 매 항에 똑같이 들어가는 공통 인자죠. 스칼라로 비유하면 $\sum_j (a\cdot b_j)\,c_j = a\sum_j b_j c_j$에서 상수 $a$를 시그마 밖으로 빼는 것과 같습니다.

벡터·행렬로 그대로 하면, Value $v_j$(길이 $d$)와 $\phi(k_j)$(길이 $m$)를 **외적**(outer product)으로 묶어 $\phi(q_i)^\top$를 밖으로 뺄 수 있습니다.

$$
o_i \;=\; \frac{\phi(q_i)^\top \Big(\sum_{j=1}^{n}\phi(k_j)\,v_j^\top\Big)}{\phi(q_i)^\top \Big(\sum_{j=1}^{n}\phi(k_j)\Big)} .
$$

> **외적 $\phi(k_j)\,v_j^\top$이 뭔가요?** 길이 $m$ 열벡터 × 길이 $d$ 행벡터 = $m\times d$ **행렬**입니다. 원소 $(a,b)$는 $\phi(k_j)_a \cdot (v_j)_b$. 토큰 $j$ 하나에 대해, "feature 차원 $m$"과 "Value 차원 $d$"를 짝지은 작은 표라고 보면 됩니다.

이제 이 표들을 전부 더한 것을 두 상태로 이름 붙입니다.

$$
S \;=\; \sum_{j=1}^{n}\phi(k_j)\,v_j^\top \;\in\; \mathbb{R}^{m\times d},
\qquad
z \;=\; \sum_{j=1}^{n}\phi(k_j)\;\in\;\mathbb{R}^{m}.
$$

$S$는 모든 토큰의 $m\times d$ 표를 하나로 합친 **누적 요약**이고, $z$는 정규화를 위한 feature들의 합입니다. 그러면 출력은

$$
\boxed{\,o_i \;=\; \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}\,}
$$

차원을 점검해 봅시다. $\phi(q_i)^\top$는 $1\times m$, $S$는 $m\times d$ → 분자 $\phi(q_i)^\top S$는 $1\times d$(출력 벡터 하나). 분모 $\phi(q_i)^\top z$는 $1\times m$과 $m\times 1$의 곱이라 스칼라. 딱 맞습니다.

**핵심은 $S$와 $z$가 $i$에 의존하지 않는다는 것**입니다. 시퀀스 전체에 대해 **딱 한 번** 만들어 두면, 모든 Query의 출력 $o_i$가 이 공용 요약만 읽습니다. $n\times n$ 행렬은 어디에도 등장하지 않죠.

계산 순서로 보면, 우리는 $(QK^\top)V$를 $Q(K^\top V)$로 **재괄호화**한 것입니다. 곱하는 순서만 바꿨는데 중간 산물의 크기가 완전히 달라집니다.

```
[softmax 경로]   Φ_Q (n×m) · Φ_Kᵀ (m×n)  →  A (n×n)   ← n×n 병목
                 A (n×n)   · V (n×d)      →  O (n×d)

[선형 경로]      Φ_Kᵀ(m×n) · V (n×d)      →  S (m×d)   ← n에 무관
                 Φ_Q (n×m) · S (m×d)      →  O (n×d)
```

- **연산량**: $S$를 만드는 데 $O(nmd)$, Query마다 읽는 데 $O(nmd)$ → **$O(nmd)$**. 보통 $m\!\sim\!d$이니 $O(nd^2)$, 즉 **$n$에 대해 선형**.
- **메모리**: 상태 $S$(크기 $m\times d$)만 유지하면 되므로 **$O(md)$**, $n$과 무관.

쉽게 말해, **Key 쪽을 미리 $S$ 하나로 요약해 두면, Query는 매번 전체를 훑지 않고 그 요약만 보면 된다** — 이게 전부입니다.

## 4. 재귀 형태 — 어텐션이 RNN이 되는 지점

지금까지는 모든 토큰을 다 봤습니다(양방향). 그런데 언어모델은 **causal**, 즉 토큰 $i$는 자기 이전 토큰($j\le i$)만 볼 수 있습니다. 이 제약을 넣으면 $S$의 합 범위가 $j\le i$로 잘리고, 자연스럽게 **점화식**이 됩니다.

$$
S_i = S_{i-1} + \phi(k_i)\,v_i^\top, \qquad
z_i = z_{i-1} + \phi(k_i), \qquad
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}.
$$

읽는 법은 간단합니다. **"지금까지의 요약 $S_{i-1}$에, 새 토큰의 표 $\phi(k_i)v_i^\top$를 더해 $S_i$로 갱신"** 하고, 그 요약을 Query $\phi(q_i)$로 읽어 $o_i$를 낸다. 갱신은 매 스텝 $O(md)$의 덧셈뿐입니다.

이 구조가 바로 **행렬 값을 hidden state로 갖는 RNN**입니다 (Katharopoulos et al., 2020, *"Transformers are RNNs"*). 상태 $S_i\in\mathbb{R}^{m\times d}$가 "기억", 덧셈이 "갱신", $\phi(q_i)^\top S_i$가 "읽기"에 해당하죠.

```
      x1        x2        x3            x_n
      │         │         │             │
   φ(k1)v1ᵀ  φ(k2)v2ᵀ  φ(k3)v3ᵀ   …   φ(kn)vnᵀ
      │         │         │             │
S0 ─►(+)─► S1 ─►(+)─► S2 ─►(+)─► S3 … ─►(+)─► S_n     (고정 크기 m×d 상태)
      │         │         │             │
   o1=φ(q1)ᵀS1  o2       o3            o_n
```

특히 **추론(생성)** 에서 차이가 극적입니다.

- **softmax**: 새 토큰을 만들 때마다 지금까지의 모든 Key·Value를 다시 봐야 합니다(KV 캐시가 계속 자람). $i$번째 스텝이 $O(id)$ → 전체 $O(n^2 d)$.
- **리니어**: 상태 $S_i$의 크기가 $m\times d$로 **고정**이라, 시퀀스가 아무리 길어져도 스텝당 비용이 $O(md)$ **상수**. 전체 $O(nmd)$.

다시 말해, **긴 문맥을 상수 메모리로 흘려보내며 생성**할 수 있게 됩니다.

## 5. Feature map $\phi$의 선택

$\phi$를 무엇으로 두느냐가 곧 각 방법의 정체입니다. "어떻게 유사도를 쪼갤 것인가"의 서로 다른 답들이죠.

- **Katharopoulos et al. (2020)**: $\phi(x)=\mathrm{elu}(x)+1$. $\mathrm{elu}(x)+1$은 항상 양수라 $\phi\ge0$ 조건을 만족합니다. $m=d$로 두는 가장 단순한 선택.
- **Performer / FAVOR+ (Choromanski et al., 2021)**: random feature로 **softmax 커널 그 자체를 근사**합니다. 핵심 항등식은

$$
\exp(q^\top k)=\mathbb{E}_{\omega\sim\mathcal N(0,I)}\big[\,\xi_\omega(q)\,\xi_\omega(k)\big],\quad
\xi_\omega(x)=\exp\!\big(\omega^\top x-\tfrac12\|x\|^2\big)
$$

즉 $\exp(q^\top k)$를 "랜덤 방향 $\omega$에 대한 기댓값"으로 다시 쓴 것입니다. 이 기댓값을 $m$개의 $\omega$로 몬테카를로 추정하면 $\phi(x)=\tfrac{1}{\sqrt m}\big[\xi_{\omega_1}(x),\dots,\xi_{\omega_m}(x)\big]$. **unbiased 추정**이라 $m$을 늘리면 진짜 softmax에 수렴하고, 양수 feature라 분산도 작습니다.
- **Linformer (Wang et al., 2020)**: 접근이 다릅니다. $\phi$로 유사도를 쪼개는 대신, **시퀀스 축 자체를 저차원으로 사영**(projection)합니다. $E,F\in\mathbb{R}^{k\times n}$으로 $K,V$를 $n\to k$($k\ll n$)로 눌러 $O(nk)$를 얻죠.

## 6. 복잡도 비교

| | 학습 연산량 | 메모리 | 추론(토큰당) |
|---|---|---|---|
| softmax attention | $O(n^2 d)$ | $O(n^2 + nd)$ | $O(nd)$ (KV 캐시↑) |
| linear attention | $O(nmd)$ | $O(md)$ (재귀) | $O(md)$ 상수 |
| Linformer | $O(nkd)$ | $O(nk)$ | — |

![복잡도 곡선: softmax O(n²) vs 리니어 O(nm)](/images/complexity_curve.png)
> 시퀀스 길이 $n$에 대한 연산량(양쪽 로그 스케일). softmax attention은 $O(n^2)$으로 위로 휘어 솟고, linear attention은 $O(nm)$으로 완만합니다. 두 곡선은 $n\approx m$에서 교차 — 짧은 시퀀스에선 오히려 softmax가 쌉니다.

로그-로그 축에서 기울기가 곧 지수입니다. softmax는 기울기 2($n^2$), 리니어는 기울기 1($n$)이라, 두 직선이 $n\approx m$에서 만나고 그 뒤로 격차가 벌어집니다. **짧은 시퀀스에선 리니어가 이득이 없다**는 점도 그래프가 정직하게 보여줍니다.

## 7. 어디서 정확도를 잃는가 — 랭크 병목

공짜 점심은 없습니다. 리니어 어텐션이 실제로 만드는 "유효 어텐션 행렬"을 써 보면

$$
\tilde A = \Phi_Q\,\Phi_K^\top,\qquad \Phi_Q,\Phi_K\in\mathbb{R}^{n\times m}
$$

입니다. 여기서 선형대수의 기본 사실 하나가 결정적입니다. **두 행렬의 곱의 rank는 작은 쪽 차원을 넘지 못합니다.** $\Phi_Q$는 $n\times m$, $\Phi_K^\top$은 $m\times n$이라, 가운데 낀 차원이 $m$이므로

$$
\mathrm{rank}(\tilde A)\le m .
$$

반면 softmax attention 행렬 $A$는 최대 rank $n$까지 가능합니다.

> **rank가 왜 중요한가요?** rank는 "이 행렬이 표현할 수 있는 독립적인 패턴의 최대 개수"입니다. $\tilde A$의 rank가 $m$으로 묶인다는 건, **아무리 토큰이 많아도 $m$가지 어텐션 패턴의 조합만** 만들 수 있다는 뜻이죠.

**즉 리니어 어텐션은 $n\times n$ 상호작용을 rank $m$짜리 요약으로 압축**합니다. $n\gg m$인 긴 시퀀스에서는 이 고정 용량의 상태 $S$가 정보 병목이 되어, 특히 **정밀한 retrieval·associative recall**(예: "아까 나온 그 숫자를 정확히 다시 불러오기") 과제에서 softmax 대비 성능이 떨어집니다. 초기 리니어 어텐션이 품질에서 손해를 본 근본 원인이 바로 이 rank 상한입니다.

## 8. 현대적 확장 — 게이팅과 청크 병렬

최근 계열은 두 방향으로 이 병목을 공략합니다.

**(a) 상태에 감쇠·gate를 넣어 "무엇을 잊을지"를 학습.** 순수 리니어 어텐션의 갱신 $S_i = S_{i-1} + \phi(k_i)v_i^\top$은 모든 과거를 **동일한 가중**으로 무한정 누적합니다. 오래된 정보가 새 정보에 안 밀리니, 고정 크기 상태가 금세 포화되죠. 그래서 "망각"을 넣습니다.

$$
\text{RetNet:}\quad S_i = \gamma\,S_{i-1} + k_i v_i^\top \;\;(\gamma\in(0,1))
$$

$\gamma$를 매 스텝 곱하니, $t$스텝 전 정보는 $\gamma^t$로 **기하급수적으로 옅어집니다**(고정 감쇠).

$$
\text{GLA:}\quad S_i = \mathrm{Diag}(\alpha_i)\,S_{i-1} + k_i v_i^\top \;\;(\alpha_i=\sigma(W x_i))
$$

GLA는 감쇠를 상수 $\gamma$가 아니라 **입력에 따라 채널별로** 정하는 gate $\alpha_i$(0~1)로 둡니다. "이 입력에서 어떤 기억 축을 유지/삭제할지"를 학습하는 셈이죠.

$$
\text{DeltaNet:}\quad S_i = S_{i-1}(I-\beta_i k_i k_i^\top) + \beta_i v_i k_i^\top
$$

DeltaNet은 delta rule로, **기존 기억에서 관련 성분을 지우고 새로 덮어씁니다**(단순 누적이 아니라 갱신). Mamba/SSM 계열도 입력 의존 감쇠를 갖는 selective state space로, 형식적으로 같은 "gated linear recurrence" 가족에 속합니다.

**(b) 청크 단위 병렬(chunkwise parallel).** 순수 재귀는 앞 스텝을 기다려야 해서 GPU에서 느립니다. 그래서 시퀀스를 크기 $C$의 청크로 나눠, 청크 **내부**는 $O(C^2)$ 이차식으로 병렬 계산하고(작으니 감당 가능), 청크 **사이**는 상태 $S$만 넘겨줍니다. 이렇게 하면 학습에서 **병렬성과 선형 복잡도를 동시에** 얻습니다. FlashLinearAttention·GLA 구현의 핵심입니다.

## 9. 결론 — 이차식 없는 시퀀스 모델의 재부상

리니어 어텐션의 본질은 "빠른 어텐션"이 아니라, **어텐션을 고정 크기 상태를 든 linear recurrence로 다시 쓰는 관점의 전환**입니다. 결합법칙 한 줄이 $O(n^2)$을 $O(n)$으로 바꾸고, 그 대가로 rank $m$의 용량 한계를 지불하며, gating이 그 한계를 다시 밀어 올립니다.

RetNet·GLA·Mamba·DeltaNet이 사실상 하나의 gated linear recurrence 가족으로 수렴하고 있다는 사실은, 트랜스포머 이후의 시퀀스 모델링이 **"softmax를 버리고 상태를 되찾는"** 방향으로 움직이고 있음을 보여줍니다.

#리니어어텐션 #트랜스포머 #선형재귀 #롱컨텍스트 #Mamba
