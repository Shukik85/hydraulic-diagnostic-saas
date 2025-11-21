/**
 * Modal components auto-registration
 * 
 * Nuxt 3 auto-imports all components from components/ directory.
 * Modal components (UCreateSystemModal, UReportGenerateModal, URunDiagnosticModal)
 * are lazy-loaded on first use - this is expected behavior.
 * 
 * No explicit registration needed.
 */
import { defineNuxtPlugin } from '#app'

export default defineNuxtPlugin(() => {
  // ✅ All UI components from components/ui/ are auto-imported by Nuxt
  // ✅ Modal components are lazy-loaded when first used (v-model trigger)
  // ✅ No explicit registration needed
  
  if (process.client && process.env.NODE_ENV === 'development') {
    console.log('✨ Modal Components: Auto-import configured')
  }
})
