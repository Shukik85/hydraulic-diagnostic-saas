export default defineNuxtPlugin(async (nuxtApp) => {
  // Пропускаем auth полностью для demo страниц
  const router = useRouter()
  const currentRoute = router.currentRoute.value
  
  if (currentRoute.path === '/demo') {
    console.log('⏭️  Skipping auth for demo')
    return
  }
  
  // Для всех остальных страниц
  try {
    const authStore = useAuthStore()
    await authStore.initialize()
  } catch (error) {
    console.error('❌ Auth init failed:', error)
  }
})
