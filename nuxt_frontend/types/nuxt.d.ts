// types/nuxt.d.ts
declare global {
  // Глобальные типы для Nuxt
  const definePageMeta: (meta: any) => void
  const useSeoMeta: (meta: any) => void
  const useApi: () => any
  const useAuthStore: () => any
  const useRouter: () => any
  const useLazyAsyncData: any
  const computed: any
  const ref: any
  const onMounted: any
  const onUnmounted: any
  const useNuxtApp: () => any
  const navigateTo: (path: string) => void
  const defineNuxtPlugin: (plugin: any) => void
  const defineStore: any
  const readonly: any
}

// Типы для API
declare module '~/types/api' {
  export interface User {
    id: number
    email: string
    name: string
    // добавьте другие поля
  }
}

export {}
