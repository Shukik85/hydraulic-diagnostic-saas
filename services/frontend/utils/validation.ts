/**
 * Form Validation Utilities
 * 
 * Type-safe validation helpers for forms
 */

export interface ValidationResult {
  valid: boolean
  errors: Record<string, string>
}

export type ValidationRule<T = any> = (value: T) => string | null

/**
 * Validate required field
 */
export function validateRequired(value: any, fieldName: string): string | null {
  if (!value || (typeof value === 'string' && value.trim() === '')) {
    return `${fieldName} обязательное поле`
  }
  return null
}

/**
 * Validate email
 */
export function validateEmail(value: string): string | null {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(value)) {
    return 'Некорректный email'
  }
  return null
}

/**
 * Validate min length
 */
export function validateMinLength(value: string, min: number): string | null {
  if (value.length < min) {
    return `Минимум ${min} символов`
  }
  return null
}

/**
 * Validate max length
 */
export function validateMaxLength(value: string, max: number): string | null {
  if (value.length > max) {
    return `Максимум ${max} символов`
  }
  return null
}

/**
 * Validate form with multiple rules
 */
export function validateForm<T extends Record<string, any>>(
  data: T,
  rules: Record<keyof T, ValidationRule[]>
): ValidationResult {
  const errors: Record<string, string> = {}
  
  for (const [field, fieldRules] of Object.entries(rules) as [keyof T, ValidationRule[]][]) {
    const value = data[field]
    
    for (const rule of fieldRules) {
      const error = rule(value)
      if (error) {
        errors[field as string] = error
        break
      }
    }
  }
  
  return {
    valid: Object.keys(errors).length === 0,
    errors
  }
}
