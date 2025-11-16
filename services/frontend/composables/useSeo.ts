/**
 * useSeo.ts - SEO meta management composable
 * Обеспечивает консистентные SEO meta tags для всех страниц
 */

import type { UseSeoMetaInput } from '@unhead/schema'

export interface SeoMetaOptions {
  title: string
  description: string
  image?: string
  url?: string
  type?: 'website' | 'article' | 'profile'
  article?: {
    publishedTime?: string
    modifiedTime?: string
    author?: string
    section?: string
    tags?: string[]
  }
}

/**
 * Базовые SEO настройки проекта
 */
const DEFAULT_SEO = {
  siteName: 'Hydraulic Diagnostic SaaS',
  siteUrl: 'https://hydraulic-diagnostic.com',
  defaultImage: 'https://hydraulic-diagnostic.com/og-default.jpg',
  twitterHandle: '@hydraulicdiag',
  locale: 'ru_RU',
  alternateLocale: 'en_US'
} as const

/**
 * Composable для управления SEO meta tags
 * 
 * @example
 * ```typescript
 * // В page компоненте
 * const { setPageMeta } = useSeo()
 * 
 * setPageMeta({
 *   title: 'Dashboard',
 *   description: 'Real-time hydraulic system monitoring',
 *   image: '/og-dashboard.jpg'
 * })
 * ```
 */
export const useSeo = () => {
  const route = useRoute()
  
  /**
   * Установить SEO meta для страницы
   */
  const setPageMeta = (options: SeoMetaOptions): void => {
    const {
      title,
      description,
      image = DEFAULT_SEO.defaultImage,
      url = `${DEFAULT_SEO.siteUrl}${route.path}`,
      type = 'website',
      article
    } = options
    
    const fullTitle = `${title} | ${DEFAULT_SEO.siteName}`
    const absoluteImage = image.startsWith('http') 
      ? image 
      : `${DEFAULT_SEO.siteUrl}${image}`
    
    const meta: UseSeoMetaInput = {
      // Basic meta
      title: fullTitle,
      description,
      
      // Open Graph
      ogTitle: fullTitle,
      ogDescription: description,
      ogType: type,
      ogUrl: url,
      ogImage: absoluteImage,
      ogSiteName: DEFAULT_SEO.siteName,
      ogLocale: DEFAULT_SEO.locale,
      ogLocaleAlternate: DEFAULT_SEO.alternateLocale,
      
      // Twitter Card
      twitterCard: 'summary_large_image',
      twitterTitle: fullTitle,
      twitterDescription: description,
      twitterImage: absoluteImage,
      twitterSite: DEFAULT_SEO.twitterHandle,
      twitterCreator: DEFAULT_SEO.twitterHandle,
    }
    
    // Article-specific meta
    if (type === 'article' && article) {
      meta.articlePublishedTime = article.publishedTime
      meta.articleModifiedTime = article.modifiedTime
      meta.articleAuthor = article.author
      meta.articleSection = article.section
      meta.articleTag = article.tags
    }
    
    useSeoMeta(meta)
  }
  
  /**
   * Установить breadcrumbs schema.org
   */
  const setBreadcrumbs = (items: Array<{ name: string; url: string }>): void => {
    const breadcrumbList = {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      'itemListElement': items.map((item, index) => ({
        '@type': 'ListItem',
        'position': index + 1,
        'name': item.name,
        'item': `${DEFAULT_SEO.siteUrl}${item.url}`
      }))
    }
    
    useHead({
      script: [
        {
          type: 'application/ld+json',
          innerHTML: JSON.stringify(breadcrumbList)
        }
      ]
    })
  }
  
  /**
   * Установить Organization schema.org
   */
  const setOrganizationSchema = (): void => {
    const organization = {
      '@context': 'https://schema.org',
      '@type': 'Organization',
      'name': DEFAULT_SEO.siteName,
      'url': DEFAULT_SEO.siteUrl,
      'logo': `${DEFAULT_SEO.siteUrl}/logo.png`,
      'sameAs': [
        'https://twitter.com/hydraulicdiag',
        'https://linkedin.com/company/hydraulicdiag',
        'https://github.com/hydraulicdiag'
      ],
      'contactPoint': {
        '@type': 'ContactPoint',
        'telephone': '+7-XXX-XXX-XXXX',
        'contactType': 'Customer Service',
        'availableLanguage': ['Russian', 'English']
      }
    }
    
    useHead({
      script: [
        {
          type: 'application/ld+json',
          innerHTML: JSON.stringify(organization)
        }
      ]
    })
  }
  
  /**
   * Установить WebSite schema.org с search action
   */
  const setWebsiteSchema = (): void => {
    const website = {
      '@context': 'https://schema.org',
      '@type': 'WebSite',
      'name': DEFAULT_SEO.siteName,
      'url': DEFAULT_SEO.siteUrl,
      'potentialAction': {
        '@type': 'SearchAction',
        'target': {
          '@type': 'EntryPoint',
          'urlTemplate': `${DEFAULT_SEO.siteUrl}/search?q={search_term_string}`
        },
        'query-input': 'required name=search_term_string'
      }
    }
    
    useHead({
      script: [
        {
          type: 'application/ld+json',
          innerHTML: JSON.stringify(website)
        }
      ]
    })
  }
  
  /**
   * Установить canonical URL
   */
  const setCanonical = (url?: string): void => {
    const canonicalUrl = url || `${DEFAULT_SEO.siteUrl}${route.path}`
    
    useHead({
      link: [
        {
          rel: 'canonical',
          href: canonicalUrl
        }
      ]
    })
  }
  
  /**
   * Установить alternate language URLs
   */
  const setAlternateLanguages = (languages: Array<{ code: string; url: string }>): void => {
    useHead({
      link: languages.map(lang => ({
        rel: 'alternate',
        hreflang: lang.code,
        href: `${DEFAULT_SEO.siteUrl}${lang.url}`
      }))
    })
  }
  
  /**
   * Установить robots meta
   */
  const setRobots = (options: {
    index?: boolean
    follow?: boolean
    noarchive?: boolean
    noimageindex?: boolean
  }): void => {
    const {
      index = true,
      follow = true,
      noarchive = false,
      noimageindex = false
    } = options
    
    const robotsValue = [
      index ? 'index' : 'noindex',
      follow ? 'follow' : 'nofollow',
      noarchive && 'noarchive',
      noimageindex && 'noimageindex'
    ].filter(Boolean).join(', ')
    
    useHead({
      meta: [
        {
          name: 'robots',
          content: robotsValue
        }
      ]
    })
  }
  
  return {
    setPageMeta,
    setBreadcrumbs,
    setOrganizationSchema,
    setWebsiteSchema,
    setCanonical,
    setAlternateLanguages,
    setRobots,
    DEFAULT_SEO
  }
}

/**
 * Preset для главной страницы
 */
export const useHomeSeo = (): void => {
  const { setPageMeta, setWebsiteSchema, setOrganizationSchema } = useSeo()
  
  setPageMeta({
    title: 'AI-Powered Hydraulic System Diagnostics',
    description: 'Advanced hydraulic system diagnostics with AI-powered predictive maintenance, real-time monitoring, and automated fault detection. Prevent failures before they happen.',
    image: '/og-home.jpg'
  })
  
  setWebsiteSchema()
  setOrganizationSchema()
}

/**
 * Preset для dashboard
 */
export const useDashboardSeo = (): void => {
  const { setPageMeta, setBreadcrumbs } = useSeo()
  
  setPageMeta({
    title: 'Dashboard',
    description: 'Real-time hydraulic system monitoring dashboard with AI-powered diagnostics, predictive maintenance, and performance analytics',
    image: '/og-dashboard.jpg'
  })
  
  setBreadcrumbs([
    { name: 'Home', url: '/' },
    { name: 'Dashboard', url: '/dashboard' }
  ])
}

/**
 * Preset для страницы системы
 */
export const useSystemSeo = (systemName: string, systemId: string): void => {
  const { setPageMeta, setBreadcrumbs } = useSeo()
  
  setPageMeta({
    title: `System ${systemName}`,
    description: `Monitor and diagnose hydraulic system ${systemName} in real-time. View performance metrics, anomaly detection, and maintenance recommendations.`,
    image: '/og-system.jpg'
  })
  
  setBreadcrumbs([
    { name: 'Home', url: '/' },
    { name: 'Systems', url: '/systems' },
    { name: systemName, url: `/systems/${systemId}` }
  ])
}
