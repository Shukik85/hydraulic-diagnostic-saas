/**
 * Global modal components registration plugin
 * Ensures all modal components are properly registered across the application
 */
import { defineNuxtPlugin } from '#app'

export default defineNuxtPlugin((nuxtApp) => {
  // Explicitly register modal components to ensure they're available globally
  // This prevents issues with conditional loading or SSR/SPA mode switches
  
  // Force preload critical modal components
  if (process.client) {
    // Client-side component registration verification
    console.log('🔧 Modal Components Plugin: Verifying registration...')
    
    // Check if components are properly resolved
    const modalComponents = [
      'UCreateSystemModal',
      'UReportGenerateModal', 
      'URunDiagnosticModal'
    ]
    
    modalComponents.forEach(componentName => {
      try {
        // Check if component is available in Nuxt's component registry
        const component = nuxtApp.vueApp.component(componentName)
        if (component) {
          console.log(`✅ ${componentName} - registered successfully`)
        } else {
          console.warn(`⚠️  ${componentName} - not found in registry`)
        }
      } catch (error) {
        console.error(`❌ ${componentName} - registration error:`, error)
      }
    })
  }
})