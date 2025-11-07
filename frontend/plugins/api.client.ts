export default defineNuxtPlugin(() => {
  const authStore = useAuthStore()

  $fetch.create({
    onRequest({ options }) {
      const headers = new Headers()

      const h = options.headers as HeadersInit | undefined
      if (h instanceof Headers) {
        h.forEach((v, k) => headers.set(k, v))
      } else if (Array.isArray(h)) {
        for (const [k, v] of h) headers.set(k, String(v))
      } else if (h && typeof h === 'object') {
        for (const [k, v] of Object.entries(h)) headers.set(k, String(v))
      }

      if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json')

      if (authStore.isAuthenticated) {
        const token = useCookie('access-token').value
        if (token) headers.set('Authorization', `Bearer ${token}`)
      }

      options.headers = headers
    },
  })
})
