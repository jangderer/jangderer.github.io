---
title: "작은 모델이 던지고 큰 모델이 받는다"
subtitle: "speculative decoding은 어떻게 '공짜로' 빨라지나"
description: "draft·verify와 rejection sampling이 출력 분포를 그대로 지키면서 속도를 얻는 원리, 그리고 EAGLE·Medusa·DSpark까지 최신 고도화."
pubDate: 2026-09-02
order: 3
track: "추론 효율"
tags: ["speculative-decoding","decode","EAGLE","Medusa","추론"]
hero: "/images/spec_speedup.png"
---

지난 편에서 우리는 **decode가 memory-bound**라는 결론에 도달했습니다. 토큰 하나를 뽑을 때마다 모델 가중치 수십~수백 GB를 통째로 메모리에서 읽어야 하고, 정작 연산 유닛은 놀죠. 그렇다면 자연스러운 질문. **"어차피 가중치를 한 번 읽을 거면, 그 한 번에 토큰을 여러 개 확정하면 안 되나?"**

바로 그 발상이 **speculative decoding**입니다. 놀라운 점은, 이게 품질을 근사하는 트릭이 **아니라는** 것입니다. 출력 분포를 **원래 모델과 수학적으로 완전히 동일**하게 유지하면서 속도만 얻습니다. 이 글에서는 그 "공짜 점심"이 어떻게 가능한지를 rejection sampling의 유도까지 따라가며 풀겠습니다. 용어는 영어 표기를 그대로 씁니다.

## 1. 문제: decode는 forward 한 번에 토큰 하나뿐

큰 **target 모델** $p$로 문장을 생성한다고 합시다. 표준 decode는 forward 한 번에 토큰 하나. 그 forward의 비용 대부분은 **가중치를 HBM에서 읽는 대역폭**에 쓰이고, 실제 산술 연산은 그 대역폭을 다 못 채웁니다(intensity ≈ 1 FLOP/byte). **한 번 읽은 가중치로 토큰 하나만 만드는 건 낭비**라는 뜻이죠.

## 2. 아이디어: draft-then-verify

핵심은 값싼 **draft 모델** $q$(예: 같은 계열의 작은 모델)를 세우는 것입니다.

1. **Draft**: $q$가 다음 $\gamma$개 토큰 $\tilde x_1,\dots,\tilde x_\gamma$을 **싸게, 순차적으로** 제안합니다.
2. **Verify**: target $p$가 이 $\gamma{+}1$개 위치를 **단 한 번의 forward로 병렬 채점**합니다. 여기가 정확히 prefill과 같은 상황 — 토큰이 여러 개니 연산 유닛이 꽉 차, compute를 제대로 씁니다.
3. **Accept/Reject**: 아래 규칙으로 앞에서부터 채택하다가, 처음 거부되는 지점에서 멈추고 그 자리를 target 분포로 고쳐 씁니다.

```
draft(q):   x̃1 → x̃2 → x̃3 → x̃4        (싸게 γ개 제안, 순차)
              │    │    │    │
verify(p):  ┌─┴────┴────┴────┴─┐
            │  단 1회 forward   │        (γ+1 위치 병렬 채점)
            └──────────────────┘
accept:      ✓    ✓    ✗                 (x̃3에서 거부 → 여기서 교정 후 중단)
결과:        x1   x2   x3'               (한 사이클에 여러 토큰 확정)
```

쉽게 말해, **작은 모델이 앞질러 여러 수를 던지고, 큰 모델이 그 수들을 한 번에 채점**합니다. 맞은 만큼 공짜로 건너뛰죠.

## 3. 정확성의 핵심 — speculative sampling

여기서 "품질 손해 없음"이 어떻게 보장될까요? **rejection sampling**입니다. 이제부터 $p(x)$와 $q(x)$는 각각 **target·draft 모델이 특정 토큰 $x$에 부여하는 확률**을 뜻합니다(앞의 $p,q$가 "모델"이었다면, 여기선 그 모델이 낸 확률값). draft가 제안한 토큰 $\tilde x \sim q(\cdot)$($q$에서 뽑은 토큰)을 이렇게 처리합니다.

$$
\text{accept } \tilde x \text{ with probability } \min\!\left(1,\ \frac{p(\tilde x)}{q(\tilde x)}\right).
$$

거부되면, 그 자리를 **잔차(residual) 분포**에서 새로 뽑습니다.

$$
p'(x) = \frac{\big(p(x)-q(x)\big)_+}{\sum_{x'}\big(p(x')-q(x')\big)_+},\qquad (y)_+ = \max(y,0).
$$

이 규칙이 왜 정확히 $p$를 재현하는지, 한 위치에서 **최종적으로 어떤 토큰 $x$가 방출될 확률** $\Pr[\text{emit }x]$을 써 보면 드러납니다. 아래에서 $\beta$는 draft 토큰이 **거부될 확률**입니다(구체적인 값은 바로 다음 줄에서 계산).

$$
\Pr[\text{emit }x] = \underbrace{q(x)\min\!\left(1,\tfrac{p(x)}{q(x)}\right)}_{\text{제안+채택}} + \underbrace{\beta\, p'(x)}_{\text{거부 후 재추출}} .
$$

첫 항은 $q(x)\min(1,p/q)=\min\big(p(x),q(x)\big)$로 깔끔히 정리됩니다. 그리고 거부 확률 $\beta=\sum_x\big(p(x)-q(x)\big)_+$가 잔차의 정규화 상수와 정확히 같아서, 둘째 항은 $\big(p(x)-q(x)\big)_+$가 됩니다. 따라서

$$
\Pr[\text{emit }x] = \min\big(p,q\big) + \big(p-q\big)_+ = p(x).
$$

마지막 등식은 경우를 나눠 보면 자명합니다: $p\le q$면 $\min=p,\ (p-q)_+=0$; $p>q$면 $\min=q,\ (p-q)_+=p-q$. 어느 쪽이든 합은 $p(x)$죠.

**즉 draft가 아무리 엉터리여도, 최종 출력 분포는 target $p$와 한 치도 다르지 않습니다.** draft의 품질은 *속도*에만 영향을 줄 뿐, *정확도*엔 영향이 없습니다. (모든 draft 토큰이 통과하면, target의 병렬 forward에서 얻은 다음 분포로 **보너스 토큰** 하나를 더 뽑아 사이클당 최대 $\gamma{+}1$개를 확정합니다.)

## 4. 얼마나 빨라지나 — acceptance rate가 전부

이득의 크기는 **acceptance rate $\alpha$**(draft 토큰이 채택될 평균 확률)가 좌우합니다. 한 사이클(=target forward 1회)에 기대되는 확정 토큰 수는 등비급수로

$$
\mathbb{E}[\text{tokens/cycle}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$

여기에 draft 비용(target 대비 비율 $c$)을 반영하면, 실제 wall-clock speedup은 (Leviathan et al., 2023)

$$
\text{speedup}(\alpha,\gamma,c) = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)\,(\gamma c + 1)} .
$$

![Speculative decoding speedup vs acceptance rate](/images/spec_speedup.png)
> draft 비용 $c=0.15$일 때, 제안 개수 $\gamma$별 기대 speedup. **$\alpha$가 지배적**입니다. $\gamma$를 키우면 $\alpha$가 높을 때만 유리하고(초록), $\alpha$가 낮으면 오히려 손해(draft만 잔뜩 하고 다 거부). 그래서 관건은 "draft를 얼마나 target과 닮게 만드느냐"입니다.

다시 말해, **속도의 열쇠는 큰 draft가 아니라 정확한 draft** — $\alpha$를 높이는 것입니다.

## 5. 지금 어디까지 왔나 — 최신 고도화 지형 (2024~2025)

목표는 둘 중 하나입니다: **$\alpha$를 높이거나, 한 번의 검증에서 더 많은 후보를 확정**하거나. 최근 연구는 네 축으로 갈립니다 — (a) draft를 어디서 얻나, (b) 검증을 어떻게 구조화하나(선형→트리), (c) 어떤 병목을 겨냥하나(가중치 vs KV cache), (d) 정적이냐 적응형이냐. **아래 방법들은 모두 rejection sampling 계열이라 출력 분포는 그대로 보존**(lossless)합니다.

**(a) 더 똑똑한 draft**

- **EAGLE 계열** — 토큰이 아니라 **feature(직전 레이어 hidden state) 수준에서 autoregression**을 수행합니다. feature 시퀀스가 더 규칙적이라 예측이 쉽다는 관찰이죠. EAGLE-1이 약 2.7–3.5x, **EAGLE-2**는 문맥 난이도에 따라 draft tree를 키우는 **dynamic tree**로 약 3–4x, **EAGLE-3**는 feature 예측을 버리고 다층 feature를 융합·"training-time test"로 학습해 **acceptance를 위치 무관 70–80%대로 평탄화**(약 3–6.5x, 코드·정형 태스크에서 상단). 2025년 들어 vLLM·SGLang·TensorRT-LLM의 **사실상 표준**입니다.
- **Medusa / Hydra** — 별도 draft 모델 없이 원 모델 위에 **여러 예측 head**를 붙여 미래 토큰들을 병렬 제안하고 **tree attention**으로 한 번에 검증(약 2.2–3.6x). Medusa head가 서로 독립적인 한계를 **Hydra**가 "앞 후보를 보도록"(sequentially dependent) 고쳐 채택 길이를 늘립니다.
- **ReDrafter (Apple)** — **RNN을 draft로** 쓰고 beam search + dynamic tree attention. **TensorRT-LLM에 정식 통합**돼 프로덕션에서 최대 2.5x — "연구 기법이 서빙 엔진 내부에 내장된" 대표 사례입니다.
- **MTP (Multi-Token Prediction)** — 별도 draft 모델도, 추가 head 학습도 없이, 애초에 **여러 미래 토큰을 예측하도록 학습된 목적함수**(DeepSeek-V3의 MTP 모듈)를 **그대로 추론 draft로 재사용**합니다 — "학습이 미리 만들어 둔 draft"인 셈이죠. DeepSeek-V3는 이 MTP로 다음 한 토큰을 미리 보는데 acceptance가 ~85–90%라 보고했고, **SGLang·vLLM·TensorRT-LLM**이 `MTP`를 spec 알고리즘으로 지원합니다(SGLang은 DeepSeek-V3에서 스텝당 2.18–2.44 토큰 채택·소규모 배포 throughput 최대 +60% 보고). 원조는 Gloeckle et al.(2024)의 **병렬** 다중 head이고, DeepSeek은 인과성을 지키는 **순차** 모듈로 변형해 씁니다.

**(b) 검증을 트리로**

단일 사슬 대신 여러 후보 **가지**(token tree)를 한 번에 검증하면 기대 채택 수가 늘어납니다. **SpecInfer**가 트리 검증(LLM을 "tree verifier"로 사용)을 정립했고, **Sequoia**는 **최적 트리 구조를 dynamic programming으로 탐색**하고 하드웨어에 맞춰 트리 크기를 자동 조정해 약 2.3–4x(A100)를 냅니다.

**(c) draft 모델 자체를 없애기**

- **Self-speculative (LayerSkip)** — **초기 레이어에서 early-exit해 draft**, 나머지 레이어로 검증·정정. 별도 모델·KV cache 없이 draft와 verify가 자원을 공유해 메모리 부담이 낮습니다.
- **Lookahead decoding** — draft를 아예 없애고, autoregression을 비선형 방정식으로 보아 **Jacobi 반복**으로 여러 n-gram을 동시에 수렴·생성합니다(exact).

**(d) 긴 컨텍스트·큰 배치로**

지난 편에서 봤듯 긴 컨텍스트·큰 배치에선 병목이 가중치가 아니라 **KV cache 로딩**입니다. **TriForce**·**MagicDec**는 sparse/compressed KV cache를 중간 draft로 삼는 **계층적 self-speculation**으로, "speculative decoding은 저지연·소배치에서만 유리하다"는 통념을 깨고 long-context·고처리량에서도 이득을 냅니다.

**그리고 프런티어 — draft의 순차성을 깬다**

지금까지의 draft는 (작은 모델이든 head든) 토큰을 대체로 **순차로** 뽑았습니다. 가장 최근의 두 방향이 이 순차성을 공략합니다.

- **DFlash** (block diffusion draft, arXiv:2602.06036, 2026) — draft를 **block diffusion 모델**로 만들어, draft 블록 전체를 **단 한 번의 forward로 병렬 생성**합니다. 다음 편에서 다룰 **Diffusion LLM의 병렬 생성**을 *drafting 단계에만* 빌려 쓰는 셈이죠(최종 품질은 target의 verify가 보장하니 lossless). 저자들은 EAGLE-3 대비 최대 2.5배, 일반 AR 대비 6배 이상 무손실 가속을 보고합니다.
- **DSpark** (arXiv:2607.05147, 2026-07) — **semi-autoregressive drafter**(병렬 backbone + 경량 sequential 모듈로 "뒤쪽 draft 토큰의 품질 저하(suffix decay)"를 완화)와 **confidence-scheduled verification**(요청별로 prefix 생존 확률·throughput을 추정해 **검증 길이를 동적으로 조절**, 여전히 lossless)을 결합합니다. 대규모 서빙에서 동일 throughput 기준 사용자당 생성 속도를 크게 끌어올렸다고 보고됩니다.

*(DFlash·DSpark 모두 집필 시점 기준 매우 최신이라, 정밀 수치·귀속은 원문(각 arXiv)을 직접 확인하길 권합니다.)*

## 6. 실무에서 기억할 것

- **배치가 크면 이득이 줄어듭니다.** continuous batching으로 이미 decode가 compute-bound에 가까워졌다면, "남는 연산"이 없어 speculative decoding이 먹을 여유가 적습니다. 그래서 **저지연 단건(batch 1~소수)** 상황에서 특히 강력합니다.
- **정확도는 보존, 지연은 분산**: 출력 분포는 정확히 같지만, 사이클마다 채택 수가 달라 **토큰 간 지연(ITL)의 분산**이 생깁니다.
- draft와 target의 **vocab이 일치**해야 $p(x)/q(x)$ 비교가 성립합니다.

## 7. 결론 — 정확도를 한 톨도 버리지 않는 가속

speculative decoding의 아름다움은 **근사가 아니라 정확한 알고리즘**이라는 데 있습니다. rejection sampling이 draft의 실수를 남김없이 교정하기에, 우리는 품질을 전혀 내주지 않고 memory-bound라는 벽을 우회합니다.

지난 편의 **prefill/decode**가 LLM 추론의 지형을 그렸다면, speculative decoding은 그 지형의 가장 단단한 벽(memory-bound decode)을 **공짜에 가깝게 통과하는 길**입니다. Medusa·EAGLE로 이어지는 흐름은, 결국 "**큰 모델을 매번 다 읽지 않고도 큰 모델의 답을 얻는다**"는 하나의 목표를 향하고 있습니다.

#speculativedecoding #LLM추론 #decode #Medusa #EAGLE #vLLM
