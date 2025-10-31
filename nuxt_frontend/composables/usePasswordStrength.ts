import type { UiPasswordStrength } from '~/types/api'

export const usePasswordStrength = (password: Ref<string>) => {
  return computed<UiPasswordStrength>(() => {
    const pwd = password.value
    
    if (!pwd || pwd.length === 0) {
      return { score: 0, label: 'weak', color: 'red' }
    }
    
    let score = 0
    let label: UiPasswordStrength['label'] = 'weak'
    let color: UiPasswordStrength['color'] = 'red'
    
    // Length check
    if (pwd.length >= 8) score += 25
    if (pwd.length >= 12) score += 25
    
    // Character variety
    if (/[a-z]/.test(pwd)) score += 10
    if (/[A-Z]/.test(pwd)) score += 15
    if (/[0-9]/.test(pwd)) score += 15
    if (/[^A-Za-z0-9]/.test(pwd)) score += 10
    
    // Assign label and color
    if (score >= 80) {
      label = 'strong'
      color = 'green'
    } else if (score >= 60) {
      label = 'good' 
      color = 'blue'
    } else if (score >= 40) {
      label = 'fair'
      color = 'yellow'
    }
    
    return { score, label, color }
  })
}