// Password strength composable with guaranteed return type
import type { PasswordStrength } from '~/types/api'

export const usePasswordStrength = (password: Ref<string>): ComputedRef<PasswordStrength> => {
  return computed(() => {
    const pwd = password.value
    
    // Default return to prevent undefined
    if (!pwd) {
      return {
        score: 0,
        label: 'Введите пароль',
        color: 'gray' as const
      }
    }
    
    let score = 0
    let label = ''
    let color: PasswordStrength['color'] = 'red'
    
    // Length check
    if (pwd.length >= 8) score += 1
    if (pwd.length >= 12) score += 1
    
    // Character variety
    if (/[a-z]/.test(pwd)) score += 1
    if (/[A-Z]/.test(pwd)) score += 1
    if (/[0-9]/.test(pwd)) score += 1
    if (/[^a-zA-Z0-9]/.test(pwd)) score += 1
    
    // Determine strength
    if (score <= 2) {
      label = 'Слабый'
      color = 'red'
    } else if (score <= 3) {
      label = 'Средний'
      color = 'yellow'
    } else if (score <= 4) {
      label = 'Хороший'
      color = 'green'
    } else {
      label = 'Отличный'
      color = 'green'
    }
    
    return { score, label, color }
  })
}