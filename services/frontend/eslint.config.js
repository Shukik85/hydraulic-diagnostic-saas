// ESLint Configuration for Hydraulic Diagnostic SaaS
// Strict TypeScript + Vue 3 + Import rules + Nuxt Auto-imports

import js from '@eslint/js';
import typescript from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import vue from 'eslint-plugin-vue';
import vueParser from 'vue-eslint-parser';
import importPlugin from 'eslint-plugin-import';
import prettier from 'eslint-config-prettier';

export default [
  // Ignore patterns
  {
    ignores: [
      'node_modules',
      '.nuxt',
      '.output',
      'dist',
      '.cache',
      'coverage',
      'cypress/videos',
      'cypress/screenshots',
      // Specific config files that don't need linting
      'tailwind.config.ts',
      'vitest.config.ts',
      'cypress.config.ts',
      'prettier.config.js',
    ],
  },

  // JavaScript/TypeScript base
  js.configs.recommended,

  // TypeScript files
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.vue'],
    plugins: {
      '@typescript-eslint': typescript,
    },
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        project: './.nuxt/tsconfig.json', // Nuxt 4 auto-generated
      },
      globals: {
        // Nuxt auto-imports (composables, utils)
        definePageMeta: 'readonly',
        defineNuxtComponent: 'readonly',
        defineNuxtPlugin: 'readonly',
        defineNuxtRouteMiddleware: 'readonly',
        navigateTo: 'readonly',
        abortNavigation: 'readonly',
        useFetch: 'readonly',
        useAsyncData: 'readonly',
        useLazyFetch: 'readonly',
        useLazyAsyncData: 'readonly',
        useNuxtApp: 'readonly',
        useRuntimeConfig: 'readonly',
        useState: 'readonly',
        useCookie: 'readonly',
        useRequestHeaders: 'readonly',
        useRequestEvent: 'readonly',
        useRouter: 'readonly',
        useRoute: 'readonly',
        useSeoMeta: 'readonly',
        useHead: 'readonly',
        useError: 'readonly',
        showError: 'readonly',
        clearError: 'readonly',
        createError: 'readonly',
        useHydration: 'readonly',
        callOnce: 'readonly',
        useId: 'readonly',
        useI18n: 'readonly',
        // Vue auto-imports
        ref: 'readonly',
        computed: 'readonly',
        reactive: 'readonly',
        readonly: 'readonly',
        watch: 'readonly',
        watchEffect: 'readonly',
        onMounted: 'readonly',
        onBeforeMount: 'readonly',
        onBeforeUnmount: 'readonly',
        onUnmounted: 'readonly',
        onUpdated: 'readonly',
        onBeforeUpdate: 'readonly',
        toRef: 'readonly',
        toRefs: 'readonly',
        unref: 'readonly',
        isRef: 'readonly',
        nextTick: 'readonly',
        provide: 'readonly',
        inject: 'readonly',
        defineProps: 'readonly',
        defineEmits: 'readonly',
        defineExpose: 'readonly',
        defineOptions: 'readonly',
        defineSlots: 'readonly',
        defineModel: 'readonly',
        withDefaults: 'readonly',
        // Custom auto-imports (composables, stores, utils)
        useAuthStore: 'readonly',
        useUiStore: 'readonly',
        useToast: 'readonly',
        useApi: 'readonly',
      },
    },
    rules: {
      // TypeScript strict rules
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/explicit-function-return-type': [
        'warn',
        {
          allowExpressions: true,
          allowTypedFunctionExpressions: true,
        },
      ],
      '@typescript-eslint/no-non-null-assertion': 'warn',
      '@typescript-eslint/strict-boolean-expressions': 'off',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/await-thenable': 'error',
      '@typescript-eslint/no-misused-promises': 'error',
      '@typescript-eslint/require-await': 'warn',
      // Disable no-undef for Nuxt auto-imports (TypeScript handles this)
      'no-undef': 'off',
    },
  },

  // Vue files
  {
    files: ['**/*.vue'],
    plugins: {
      vue,
    },
    languageOptions: {
      parser: vueParser,
      parserOptions: {
        parser: typescriptParser,
        ecmaVersion: 'latest',
        sourceType: 'module',
        extraFileExtensions: ['.vue'],
      },
    },
    rules: {
      ...vue.configs['vue3-recommended'].rules,

      // Vue 3 specific
      'vue/multi-word-component-names': 'off',
      'vue/no-v-html': 'warn',
      'vue/require-default-prop': 'error',
      'vue/require-prop-types': 'error',
      'vue/component-name-in-template-casing': ['error', 'PascalCase'],
      'vue/block-lang': ['error', { script: { lang: 'ts' } }],
      'vue/define-macros-order': [
        'error',
        {
          order: ['defineOptions', 'defineProps', 'defineEmits', 'defineSlots'],
        },
      ],

      // Accessibility
      'vue/html-button-has-type': 'error',
      'vue/no-static-inline-styles': 'warn',
      'vue/prefer-true-attribute-shorthand': 'error',

      // Performance
      'vue/no-setup-props-destructure': 'error',
      'vue/no-ref-object-destructure': 'error',
    },
  },

  // Import rules
  {
    plugins: {
      import: importPlugin,
    },
    rules: {
      'import/order': [
        'error',
        {
          groups: ['builtin', 'external', 'internal', ['parent', 'sibling'], 'index', 'type'],
          'newlines-between': 'always',
          alphabetize: { order: 'asc', caseInsensitive: true },
        },
      ],
      'import/no-duplicates': 'error',
      'import/no-unresolved': 'off', // Handled by TypeScript
      'import/named': 'off', // Handled by TypeScript
    },
  },

  // Prettier compatibility (must be last)
  prettier,
];
