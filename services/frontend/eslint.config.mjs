// Plain ESM JS (no TS assertions here)
let withNuxt = () => []

try {
  if (
    typeof process !== 'undefined' &&
    process.env.NODE_ENV !== 'test' &&
    !process.argv.some((a) => a.includes('typecheck'))
  ) {
    const { existsSync } = await import('fs')
    if (existsSync('./.nuxt/eslint.config.mjs')) {
      const nuxtEslint = await import('./.nuxt/eslint.config.mjs')
      // Support both default and named export without TS assertions
      withNuxt = (nuxtEslint && nuxtEslint.default) ? nuxtEslint.default : nuxtEslint
      if (typeof withNuxt !== 'function') withNuxt = () => []
    }
  }
} catch {}

export default withNuxt({
  rules: {
    // Console usage - allow only warnings and errors in production
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    
    // Vue specific rules
    'vue/multi-word-component-names': 'error',
    'vue/no-unused-components': 'error',
    'vue/component-name-in-template-casing': ['error', 'PascalCase', {
      registeredComponentsOnly: false,
      ignores: [],
    }],
    'vue/no-v-html': 'warn',
    'vue/require-default-prop': 'warn',
    'vue/require-prop-types': 'error',
    
    // TypeScript rules
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unused-vars': ['error', {
      argsIgnorePattern: '^_',
      varsIgnorePattern: '^_',
    }],
    
    // General code quality
    'prefer-const': 'error',
    'no-var': 'error',
    'eqeqeq': ['error', 'always'],
  },
})
