---
title: "Key와 Value는 헤드마다 다를 필요가 없다"
subtitle: "MQA와 GQA — KV cache를 8분의 1로 줄이면서 품질은 지키는 헤드 공유의 산술"
description: "디코딩은 KV cache 읽기에 묶인 memory-bound 연산이다. query 헤드는 두고 key·value 헤드만 줄이면 왜 arithmetic intensity가 H배 오르는지, MQA의 손실을 GQA가 어떻게 메우는지, uptraining과 실제 모델 설정까지 수식으로 정리한다."
pubDate: 2026-09-03
order: 9
track: "아키텍처"
tags: ["GQA","MQA","KV cache","attention","아키텍처"]
hero: "/images/gqa_kv.png"
---

prefill/decode 편의 결론은 "디코딩은 연산이 아니라 **메모리 읽기**에 묶인다"였습니다. 토큰 하나를 만들 때마다 지금까지의 모든 key·value(KV cache)를 HBM에서 읽어 와야 하니까요. 그렇다면 답은 간단합니다. **읽을 것을 줄이면 됩니다.**

이 글은 가장 단순하고 가장 널리 쓰이는 방법 — key·value **헤드 수를 줄이는** MQA와 GQA — 를 다룹니다. 왜 헤드를 줄여도 되는지, 줄이면 정확히 무엇이 몇 배 좋아지는지, 그리고 품질은 어떻게 지키는지를 산술로 따라갑니다. 용어는 영어를 그대로 씁니다.

## 1. 복습 — multi-head attention의 KV cache

토큰 표현의 행렬을 $X\in\mathbb{R}^{L\times d}$($L$=토큰 수, $d$=모델 차원)라 하면, 헤드 $h$($h=1,\dots,H$)의 query·key·value는

$$
Q_h = XW_h^{Q},\qquad K_h = XW_h^{K},\qquad V_h = XW_h^{V},\qquad
W_h^{Q},W_h^{K},W_h^{V}\in\mathbb{R}^{d\times d_h},
$$

$H$는 헤드 수, $d_h$는 헤드 차원(보통 $d=H\,d_h$), 각 $Q_h,K_h,V_h\in\mathbb{R}^{L\times d_h}$입니다. 헤드마다 어텐션 $\mathrm{softmax}(Q_hK_h^{\top}/\sqrt{d_h})V_h$를 계산해 이어 붙입니다.

디코딩 때는 과거 토큰의 $K_h,V_h$를 다시 계산하지 않도록 **저장**해 둡니다. 그 저장분이 KV cache이고, 크기는

$$
\text{KV cache} = \underbrace{2}_{K,V}\cdot H\cdot d_h\cdot L\cdot \ell\cdot b\ \text{ bytes},
$$

$\ell$은 층 수, $b$는 원소당 바이트(bf16=2)입니다. 70B급 모델($\ell=80,\ H=64,\ d_h=128$)이 128K 컨텍스트를 들고 있으면 $2\cdot64\cdot128\cdot128\text{K}\cdot80\cdot2\approx 335\text{GB}$. **H100 네 장 분량의 메모리가 파라미터가 아니라 캐시**에 쓰입니다.

## 2. 왜 이것이 속도 문제인가 — arithmetic intensity로

디코딩 한 스텝에서 헤드 하나가 하는 일은 새 query $q\in\mathbb{R}^{d_h}$ 하나로 $L$개의 key와 내적하고($qK^{\top}$), 그 가중치로 $L$개의 value를 합치는 것($pV$)입니다. FLOPs는 곱셈-덧셈을 2로 세어

$$
\text{FLOPs}_{\text{head}} = \underbrace{2Ld_h}_{qK^\top} + \underbrace{2Ld_h}_{pV} = 4Ld_h ,
$$

읽어야 할 메모리는 $K_h,V_h$ 합쳐 $2Ld_h$개 원소. **arithmetic intensity**(읽은 원소 1개당 연산 수)는

$$
I_{\text{MHA}} = \frac{H\cdot 4Ld_h}{H\cdot 2Ld_h} = 2\ \text{FLOPs/element}.
$$

GPU가 연산으로 바빠지려면 이 값이 수백은 되어야 하니(roofline 편), 어텐션 디코딩은 **압도적으로 memory-bound**입니다. 연산기는 놀고 HBM만 바쁩니다. 그리고 이 2라는 숫자는 $L$에도 $d_h$에도 무관합니다 — 어텐션 구조 자체의 성질입니다.

## 3. MQA — key·value를 단 하나만

Multi-Query Attention(Shazeer, 2019)의 처방: **query 헤드는 $H$개 그대로 두고, key·value는 모든 헤드가 하나를 공유**합니다.

$$
Q_h = XW_h^{Q}\ (h=1..H),\qquad K = XW^{K},\qquad V = XW^{V}.
$$

헤드 $h$의 어텐션은 $\mathrm{softmax}(Q_hK^{\top}/\sqrt{d_h})\,V$ — query만 헤드별로 다르고 key·value는 같습니다. 이제 산술이 바뀝니다.

- **KV cache**: $2\cdot H\cdot d_h$ → $2\cdot d_h$ per token per layer. **$H$배 감소.**
- **FLOPs**: 그대로 $H\cdot 4Ld_h$ (query는 여전히 $H$개).
- **읽기**: $2Ld_h$ (한 번만 읽어 $H$개 query가 재사용).

$$
I_{\text{MQA}} = \frac{H\cdot 4Ld_h}{2Ld_h} = 2H\ \text{FLOPs/element}.
$$

**arithmetic intensity가 $H$배** 오릅니다. $H=64$면 2에서 128로 — memory-bound였던 연산이 compute-bound 쪽으로 성큼 다가갑니다. 쉽게 말해, 같은 자료를 64명이 각자 복사해 읽던 것을 **한 부만 읽고 64명이 돌려 보는** 겁니다. 같은 계산을 하는데 도서관 왕복이 64분의 1이 됩니다.

대가는 **품질**입니다. key·value가 하나뿐이면 각 헤드가 "무엇을 어디서 찾을지"의 자유도가 줄어듭니다. Shazeer의 원논문과 이후 보고들에서 MQA는 MHA 대비 perplexity·downstream 점수가 약간 떨어지고, 학습이 불안정한 경우가 있었습니다.

## 4. GQA — 그 사이 어딘가

Grouped-Query Attention(Ainslie et al., 2023)은 두 극단 사이를 **하나의 손잡이**로 잇습니다. query 헤드 $H$개를 $G$개 **그룹**으로 나누고, 그룹마다 key·value 하나를 공유합니다.

$$
K_g = XW_g^{K},\quad V_g = XW_g^{V}\quad (g=1..G),\qquad
\text{헤드 } h \text{는 그룹 } g(h)=\Big\lceil \tfrac{h}{H/G}\Big\rceil\text{의 } K_{g(h)},V_{g(h)}\text{를 사용}.
$$

$G=H$면 MHA, $G=1$이면 MQA입니다. 그 사이에서

$$
\text{KV cache} \propto G,\qquad I_{\text{GQA}} = \frac{H\cdot 4Ld_h}{G\cdot 2Ld_h} = \frac{2H}{G}.
$$

![헤드 공유 구조와 KV cache](/images/gqa_kv.png)
> 왼쪽: query 헤드 8개를 몇 개의 key/value 헤드가 받치는가. MHA는 1:1, GQA($G{=}2$)는 4개 query가 하나의 K/V를 공유, MQA는 8개 모두 하나를 공유합니다. 오른쪽: 70B급(80층, $d_h{=}128$, bf16) 모델의 시퀀스당 KV cache. MHA(64 KV 헤드)는 128K에서 335GB로 H100 네 장 분량이지만, GQA-8은 42GB, MQA는 5GB입니다. 실제 Llama-2/3-70B가 GQA-8을 택한 이유입니다.

$G=8$이면 KV cache는 MHA의 1/8, arithmetic intensity는 8배. 그리고 논문의 핵심 발견은 **$G=8$ 정도에서 품질이 MHA와 거의 같다**는 것입니다(T5-XXL 기준 MHA 47.2 vs GQA-8 47.1 vs MQA 46.6, 논문 Table 1 요약). 왜일까요? MHA의 헤드별 key·value에는 **중복이 많습니다.** 여러 헤드가 비슷한 곳을 비슷한 방식으로 보고 있어서, 그룹으로 묶어도 잃는 정보가 적습니다. 반면 MQA의 "하나"는 너무 적었던 것이죠.

## 5. 이미 학습된 모델을 GQA로 — uptraining

GQA 논문의 실용적 기여 하나 더. 처음부터 GQA로 학습하지 않아도, **MHA 체크포인트를 GQA로 변환**할 수 있습니다. 그룹 $g$의 key 사영 행렬을 그 그룹 헤드들의 **평균**으로 초기화합니다.

$$
W_g^{K} = \frac{G}{H}\sum_{h:\,g(h)=g} W_h^{K},\qquad W_g^{V} = \frac{G}{H}\sum_{h:\,g(h)=g} W_h^{V}.
$$

$G/H$는 그룹 안 헤드 수의 역수(즉 산술평균)입니다. 이렇게 초기화하면 변환 직후 품질이 크게 무너지지 않고, 원래 학습량의 **약 5%** 만 더 학습(uptrain)하면 회복됩니다. 평균이 잘 통하는 이유도 §4와 같습니다 — 헤드들의 key가 원래 비슷해서, 평균이 그럭저럭 대표합니다.

## 6. 실제 모델들의 선택

| 모델 | $H$ (query) | $G$ (KV) | 비율 $H/G$ |
|---|---|---|---|
| Llama-2-70B | 64 | 8 | 8 |
| Llama-3-8B / 70B | 32 / 64 | 8 / 8 | 4 / 8 |
| Mistral-7B | 32 | 8 | 4 |
| Qwen2.5-72B | 64 | 8 | 8 |
| Gemma-2-27B | 32 | 16 | 2 |
| PaLM-540B, Falcon-7B | — | 1 (MQA) | $H$ |

2023년 이후 공개된 대형 모델은 거의 예외 없이 GQA이고, $G=8$이 사실상 표준입니다. 8은 우연이 아닙니다 — tensor parallel로 모델을 8장의 GPU에 나눌 때 **GPU 하나에 KV 헤드 하나**가 떨어지도록 맞춘 숫자입니다. 하드웨어가 아키텍처를 정한 예입니다.

## 7. 한계와 그 다음

GQA는 KV cache를 줄이는 가장 값싼 방법이지만, **줄일 수 있는 한계가 $G\ge1$** 입니다. MQA(G=1)에서 더 줄이려면 헤드 차원 $d_h$를 건드리거나 다른 축을 찾아야 합니다. 그래서 그 다음 세대는 "헤드를 줄이는" 대신 **key·value 자체를 저차원 잠재(latent)로 압축**합니다 — 헤드는 128개를 그대로 유지하면서 캐시는 MQA 수준까지 줄이는 DeepSeek의 Multi-head Latent Attention이 다음 편입니다. 그 외에도 층 간 KV 공유(Cross-Layer Attention), KV 양자화(int8/FP4)가 같은 목표를 다른 축에서 공략합니다.

## 8. 결론

MQA와 GQA는 "**key·value는 헤드마다 다를 필요가 없다**"는 관찰 하나로 KV cache와 HBM 읽기를 $H/G$배 줄였고, 그만큼 arithmetic intensity를 올려 디코딩을 빠르게 만들었습니다. 수식은 §2–4의 나눗셈 몇 줄이 전부지만, 그 나눗셈이 70B 모델을 128K 컨텍스트로 서빙할 수 있게 만든 실질적 차이입니다.

*(참고: MQA — Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150 · GQA — Ainslie et al., arXiv:2305.13245 · 모델별 $H,G$는 각 공개 config 기준. 그림은 직접 그린 것이며 KV cache 수치는 본문 식의 계산값입니다.)*

#GQA #MQA #KVcache #attention #아키텍처
