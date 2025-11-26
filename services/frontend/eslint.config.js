import { createConfigForNuxt } from '@nuxt/eslint-config/flat';
import prettier from 'eslint-plugin-prettier';
import eslintConfigPrettier from 'eslint-config-prettier';

export default createConfigForNuxt({
  features: {
    tooling: true,
    stylistic: true,
  },
  dirs: {
    src: ['./'],
  },
})
  .append({
    plugins: {
      prettier,
    },
    rules: {
      // TypeScript Rules
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/explicit-function-return-type': 'warn',
      '@typescript-eslint/no-non-null-assertion': 'warn',
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],

      // Vue Rules
      'vue/require-explicit-emits': 'error',
      'vue/no-unused-refs': 'warn',
      'vue/component-api-style': ['error', ['script-setup']],
      'vue/block-order': [
        'error',
        {
          order: ['script', 'template', 'style'],
        },
      ],
      'vue/html-self-closing': [
        'error',
        {
          html: {
            void: 'always',
            normal: 'always',
            component: 'always',
          },
        },
      ],
      'vue/multi-word-component-names': 'off',

      // General Best Practices
      'curly': ['error', 'all'],
      'no-duplicate-imports': 'error',
      'require-await': 'warn',
      'no-console': [
        'warn',
        {
          allow: ['warn', 'error'],
        },
      ],

      // Prettier integration
      'prettier/prettier': 'error',
    },
  })
  .append(eslintConfigPrettier);
