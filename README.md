# LLM 딥다이브 — 정적 블로그

수식으로 파고드는 LLM 해설 시리즈. Astro + KaTeX. velog와 병행(크로스포스트)하며, 이 사이트가 디자인·데이터를 소유하는 메인 아카이브입니다.

## 개발

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # dist/ 로 정적 빌드
npm run preview  # 빌드 결과 미리보기
```

Node 18+ 필요.

## 글 추가하는 법

`src/content/posts/` 에 마크다운 파일 하나를 추가하면 끝입니다. 프론트매터:

```markdown
---
title: "제목"
subtitle: "부제 (선택)"
description: "한 줄 소개 — 목록 카드와 메타에 쓰임"
pubDate: 2026-09-10
order: 4                 # 목록·이전/다음 정렬 순서
track: "추론 효율"        # 아키텍처 / 추론 효율 / 롱컨텍스트 / 정렬·추론 / 파격
tags: ["FlashAttention","attention"]
hero: "/images/xxx.png"  # 카드 썸네일 (선택)
---

본문. 수식은 `$인라인$` 과 `$$디스플레이$$` 로 쓰면 KaTeX가 렌더합니다.
그림은 `/public/images/` 에 넣고 `![캡션](/images/xxx.png)` 로 참조.
이미지 바로 다음 줄의 `> 인용문` 은 캡션으로 스타일됩니다.
```

수식 문법은 velog(KaTeX)와 동일하므로, velog 원고를 그대로 붙여 쓰면 됩니다.

## 그림 만들기

`scripts/` 에 matplotlib 플롯 스크립트를 두고 PNG를 `public/images/` 로 내보내는 방식을 권장합니다(라벨은 영문, 캡션은 본문 마크다운에서 한글).

## 배포

`astro.config.mjs` 의 `site` 를 본인 도메인/URL로 바꾼 뒤:

- **Vercel / Netlify (권장, 제로 설정)**: 이 저장소를 연결하면 자동 빌드. 프레임워크 프리셋 "Astro" 선택.
- **GitHub Pages (프로젝트 페이지)**: `astro.config.mjs` 에 `base: '/저장소이름'` 추가 + GitHub Actions 워크플로(`withastro/action`) 사용. `site` 는 `https://<user>.github.io` 로.

## velog 병행

각 글은 이 사이트가 원본(canonical)입니다. velog에는 동일 마크다운을 붙여 크로스포스트해 한국 개발자 독자 유입을 챙기고, 글 하단에 원본 링크를 달아두면 좋습니다.

## 구조

```
src/
  content/posts/*.md      글 (마크다운 + 프론트매터)
  content/config.ts       글 스키마
  layouts/Base.astro      공통 셸(nav·footer·테마·폰트·KaTeX)
  pages/index.astro       허브 (발행글 + 로드맵)
  pages/posts/[...slug].astro   글 렌더
  styles/global.css       디자인 토큰 + 아티클 prose
public/images/            그림
```
