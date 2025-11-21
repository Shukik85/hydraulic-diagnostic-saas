import type { UseSeoMetaInput } from '@unhead/vue'

export interface ArticleSeo {
  title: string
  description: string
  author?: string
  publishedTime?: string
  modifiedTime?: string
  section?: string
  tags?: string[]
}

export function useArticleSeo(article: ArticleSeo) {
  const meta: UseSeoMetaInput = {
    title: article.title,
    description: article.description,
    ogTitle: article.title,
    ogDescription: article.description,
    ogType: 'article',
  }

  if (article.author) {
    // articleAuthor expects array
    (meta as any).articleAuthor = [article.author]
  }

  if (article.publishedTime) {
    meta.articlePublishedTime = article.publishedTime
  }

  if (article.modifiedTime) {
    meta.articleModifiedTime = article.modifiedTime
  }

  if (article.section) {
    meta.articleSection = article.section
  }

  if (article.tags?.length) {
    meta.articleTag = article.tags
  }

  useSeoMeta(meta)
}
