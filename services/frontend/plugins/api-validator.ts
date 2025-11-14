// services/frontend/plugins/api-validator.ts
/**
 * Runtime API validation против OpenAPI schema.
 * Активен только в development mode.
 */
import Ajv from 'ajv'
import addFormats from 'ajv-formats'

export default defineNuxtPlugin((nuxtApp) => {
  // Only in development
  if (process.env.NODE_ENV !== 'development') {
    return
  }
  
  const config = useRuntimeConfig()
  
  // Skip if validation disabled
  if (config.public.disableApiValidation) {
    console.warn('⚠️  API validation disabled')
    return
  }
  
  console.log('🔍 API validation enabled (development mode)')
  
  // Load OpenAPI spec
  let openApiSpec: any
  try {
    openApiSpec = require('~/generated/openapi.json')
  } catch (error) {
    console.warn('⚠️  OpenAPI spec not found, validation disabled')
    return
  }
  
  // Initialize Ajv
  const ajv = new Ajv({
    allErrors: true,
    verbose: true,
    strict: false
  })
  addFormats(ajv)
  
  // Compile validators
  const validators = new Map<string, any>()
  
  for (const [path, methods] of Object.entries(openApiSpec.paths || {})) {
    for (const [method, spec] of Object.entries(methods as any)) {
      const operationId = spec.operationId || `${method}_${path}`
      
      // Request body validator
      if (spec.requestBody?.content?.['application/json']?.schema) {
        const schema = spec.requestBody.content['application/json'].schema
        validators.set(`${operationId}:request`, ajv.compile(schema))
      }
      
      // Response validator (200 OK)
      if (spec.responses?.['200']?.content?.['application/json']?.schema) {
        const schema = spec.responses['200'].content['application/json'].schema
        validators.set(`${operationId}:response`, ajv.compile(schema))
      }
    }
  }
  
  console.log(`✅ Compiled ${validators.size} validators`)
  
  // Intercept $fetch calls
  const originalFetch = globalThis.$fetch
  
  globalThis.$fetch = new Proxy(originalFetch, {
    apply: async (target, thisArg, args) => {
      const [url, options = {}] = args
      
      // Extract operation ID from URL
      const operationId = extractOperationId(url, options.method)
      
      // Validate request body
      if (options.body && validators.has(`${operationId}:request`)) {
        const validator = validators.get(`${operationId}:request`)
        const valid = validator(options.body)
        
        if (!valid) {
          console.error('❌ Request validation failed:', validator.errors)
          console.error('   URL:', url)
          console.error('   Body:', options.body)
          
          // Optionally throw error
          if (config.public.strictValidation) {
            throw new Error(`Invalid request body for ${operationId}`)
          }
        } else {
          console.log('✅ Request validated:', operationId)
        }
      }
      
      // Execute request
      const response = await Reflect.apply(target, thisArg, args)
      
      // Validate response
      if (validators.has(`${operationId}:response`)) {
        const validator = validators.get(`${operationId}:response`)
        const valid = validator(response)
        
        if (!valid) {
          console.error('❌ Response validation failed:', validator.errors)
          console.error('   URL:', url)
          console.error('   Response:', response)
          
          // Log to monitoring
          logValidationError(operationId, validator.errors)
        } else {
          console.log('✅ Response validated:', operationId)
        }
      }
      
      return response
    }
  })
  
  // Provide validators for manual use
  nuxtApp.provide('apiValidators', validators)
})

/**
 * Extract operation ID from URL and method.
 */
function extractOperationId(url: string, method: string = 'GET'): string {
  // Simple mapping - можно улучшить
  const path = url.replace(/^.*\/api\/v1/, '')
  return `${method.toLowerCase()}${path.replace(/\//g, '_')}`
}

/**
 * Log validation error to monitoring.
 */
function logValidationError(operationId: string, errors: any[]) {
  // Send to monitoring service
  if (process.client) {
    console.warn(`Validation error logged: ${operationId}`, errors)
    // TODO: Send to Sentry/DataDog
  }
}
