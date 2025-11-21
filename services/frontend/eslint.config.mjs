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

export default [
  // ✨ Игнорирование сгенерированных файлов (первым элементом!)
  {
    ignores: [
      '.nuxt/**',
      '.output/**',
      'dist/**',
      'node_modules/**',
      'generated/**',
      '.turbo/**',
      'coverage/**',
      'test-results/**',
      'playwright-report/**'
    ]
  },
  ...withNuxt({
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
      'vue/require-explicit-emits': 'error',
      'vue/no-unused-refs': 'warn',
      'vue/padding-line-between-blocks': 'warn',
      'vue/component-api-style': ['error', ['script-setup']],
      'vue/block-order': ['error', {
        order: ['script', 'template', 'style']
      }],
      'vue/html-self-closing': ['error', {
        html: {
          void: 'always',
          normal: 'never',
          component: 'always'
        }
      }],
      
      // TypeScript rules
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': ['error', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
      }],
      '@typescript-eslint/explicit-function-return-type': ['warn', {
        allowExpressions: true,
        allowTypedFunctionExpressions: true,
        allowHigherOrderFunctions: true
      }],
      '@typescript-eslint/no-non-null-assertion': 'warn',
      '@typescript-eslint/consistent-type-imports': ['error', {
        prefer: 'type-imports',
        fixStyle: 'inline-type-imports'
      }],
      
      // General code quality
      'prefer-const': 'error',
      'no-var': 'error',
      'eqeqeq': ['error', 'always', { null: 'ignore' }],
      'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
      'no-unused-expressions': 'error',
      'no-duplicate-imports': 'error',
      
      // Best practices
      'curly': ['error', 'all'],
      'default-case': 'warn',
      'dot-notation': 'warn',
      'no-empty-function': ['warn', { allow: ['arrowFunctions'] }],
      'no-implicit-coercion': 'error',
      'no-return-await': 'error',
      'require-await': 'warn',
    },
  })
]
