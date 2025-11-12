// Auth store with demo mode support
import { defineStore } from 'pinia'

interface User {
  id: number
  name: string
  email: string
  role: string
  avatar?: string
}

export const useAuthStore = defineStore('auth', {
  state: () => ({
    isAuthenticated: false,
    isDemoMode: false,
    user: null as User | null,
    token: null as string | null,
  }),

  getters: {
    isDemo: (state) => state.isDemoMode,
    canEdit: (state) => !state.isDemoMode, // Read-only in demo mode
  },

  actions: {
    // ✅ Demo login
    loginAsDemo() {
      const config = useRuntimeConfig()
      
      this.isAuthenticated = true
      this.isDemoMode = true
      this.user = {
        id: 0,
        name: config.public.demoUserName || 'Demo User',
        email: config.public.demoUserEmail || 'demo@hydraulic-ai.com',
        role: 'demo',
        avatar: '/img/demo-avatar.png'
      }
      this.token = 'demo-token-readonly'
      
      console.log('🎭 Logged in as DEMO user')
    },

    // ✅ Real login
    async login(email: string, password: string) {
      const config = useRuntimeConfig()
      
      // In demo mode - instant success
      if (config.public.demoMode) {
        this.loginAsDemo()
        return { success: true }
      }

      // In dev mode - mock login
      if (process.dev) {
        this.isAuthenticated = true
        this.isDemoMode = false
        this.user = {
          id: 1,
          name: 'Dev User',
          email: email,
          role: 'admin'
        }
        this.token = 'dev-token'
        return { success: true }
      }

      // Production - real API call
      try {
        const response = await $fetch(`${config.public.apiBase}/auth/login`, {
          method: 'POST',
          body: { email, password }
        }) as any
        
        this.isAuthenticated = true
        this.isDemoMode = false
        this.user = response.user
        this.token = response.token
        
        return { success: true }
      } catch (error: any) {
        return { 
          success: false, 
          error: error.message || 'Login failed' 
        }
      }
    },

    async logout() {
      if (this.isDemoMode) {
        // In demo mode - just redirect to home
        this.isAuthenticated = false
        this.isDemoMode = false
        this.user = null
        this.token = null
        navigateTo('/')
        return
      }

      if (process.dev) {
        console.log('Dev logout')
        this.isAuthenticated = false
        this.user = null
        this.token = null
        navigateTo('/auth/login')
        return
      }

      // Production logout
      const config = useRuntimeConfig()
      try {
        await $fetch(`${config.public.apiBase}/auth/logout`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${this.token}`
          }
        })
      } catch (error) {
        console.error('Logout error:', error)
      }
      
      this.isAuthenticated = false
      this.user = null
      this.token = null
      navigateTo('/auth/login')
    }
  }
})
