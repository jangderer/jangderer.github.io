import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// 배포 전 site 값을 본인 도메인으로 바꾸세요.
// GitHub Pages(프로젝트 페이지)면 base도 함께 설정: base: '/llm-blog'
export default defineConfig({
  site: 'https://example.com',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: { theme: 'github-dark', wrap: true },
  },
});
