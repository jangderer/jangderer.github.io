import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    description: z.string(),
    pubDate: z.coerce.date(),
    order: z.number(),
    track: z.string(),
    tags: z.array(z.string()).default([]),
    hero: z.string().optional(),
  }),
});

export const collections = { posts };
