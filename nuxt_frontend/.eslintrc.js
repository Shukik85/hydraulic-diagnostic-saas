module.exports = {
  root: true,
  env: {
    browser: true,
    node: true,
    es2023: true,
  },
  parser: 'vue-eslint-parser',
  parserOptions: {
    parser: '@typescript-eslint/parser',
    ecmaVersion: 2023,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  extends: [
    '@nuxt/eslint-config',
    'plugin:vue/vue3-essential',
    'plugin:@typescript-eslint/recommended',
    'plugin:prettier/recommended',
  ],
  plugins: [
    'vue',
    '@typescript-eslint',
    'prettier',
  ],
  rules: {
    // Vue specific rules
    'vue/html-self-closing': [
      'error',
      {
        html: {
          void: 'always',
          normal: 'always',
          component: 'always',
        },
        svg: 'always',
        math: 'always',
      },
    ],
    'vue/component-name-in-template-casing': ['error', 'PascalCase'],
    'vue/no-multiple-template-root': 'off',
    'vue/multi-word-component-names': 'off',
    
    // TypeScript specific rules
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/prefer-const': 'error',
    
    // General rules
    'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
    'prefer-const': 'error',
    'no-var': 'error',
    'object-shorthand': 'error',
    'prefer-template': 'error',
    
    // Prettier integration
    'prettier/prettier': 'error',
  },
  ignorePatterns: [
    'node_modules/',
    '.nuxt/',
    '.output/',
    'dist/',
    '*.min.js',
  ],
  globals: {
    $fetch: 'readonly',
    defineNuxtConfig: 'readonly',
    navigateTo: 'readonly',
    useHead: 'readonly',
    useMeta: 'readonly',
    useRoute: 'readonly',
    useRouter: 'readonly',
    useState: 'readonly',
    useCookie: 'readonly',
    useRuntimeConfig: 'readonly',
    useFetch: 'readonly',
    useLazyFetch: 'readonly',
    refreshCookie: 'readonly',
    addRouteMiddleware: 'readonly',
    definePageMeta: 'readonly',
    defineNuxtMiddleware: 'readonly',
    defineNuxtPlugin: 'readonly',
    useNuxtApp: 'readonly',
    abortNavigation: 'readonly',
    clearError: 'readonly',
    createError: 'readonly',
    isPrerendered: 'readonly',
    throwError: 'readonly',
    useError: 'readonly',
    useRequestHeaders: 'readonly',
    setResponseStatus: 'readonly',
  },
};
