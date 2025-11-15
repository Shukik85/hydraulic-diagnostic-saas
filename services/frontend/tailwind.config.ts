import type { Config } from 'tailwindcss'

/**
 * HYDRAULIC DIAGNOSTIC SAAS - TAILWIND CONFIG
 * Metallic Industrial Theme v1.0
 *
 * Unified design system synchronized with Django Admin
 * Features: Brushed metal textures, inset shadows, industrial gradients
 */

export default {
  content: [
    './components/**/*.{js,vue,ts}',
    './layouts/**/*.vue',
    './pages/**/*.vue',
    './plugins/**/*.{js,ts}',
    './app.vue',
    './error.vue',
    './stores/**/*.{js,ts}',
    './composables/**/*.{js,ts}',
    './styles/**/*.css',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Metallic base colors - промышленные металлические тона
        metal: {
          dark: '#1a1d23',
          medium: '#2d3139',
          light: '#4a4f5c',
          highlight: '#6b7280',
        },

        // Steel accents - стальные акценты
        steel: {
          dark: '#374151',
          medium: '#4b5563',
          light: '#6b7280',
          shine: '#9ca3af',
        },

        // Primary - Индиго (промышленный, приглушенный)
        primary: {
          50: '#e0e7ff',
          100: '#c7d2fe',
          200: '#a5b4fc',
          300: '#818cf8',
          400: '#6366f1',
          500: '#4f46e5', // Changed from #3b82f6 - more industrial
          600: '#4338ca',
          700: '#3730a3',
          800: '#312e81',
          900: '#1e1b4b',
          950: '#1e1b4b',
        },

        // Status colors - приглушенные промышленные
        status: {
          success: {
            DEFAULT: '#047857', // Darker green
            light: '#6ee7b7',
            dark: '#065f46',
          },
          warning: {
            DEFAULT: '#b45309', // Darker amber
            light: '#fcd34d',
            dark: '#92400e',
          },
          error: {
            DEFAULT: '#991b1b', // Darker red
            light: '#fca5a5',
            dark: '#7f1d1d',
          },
          info: {
            DEFAULT: '#075985', // Darker sky
            light: '#7dd3fc',
            dark: '#0c4a6e',
          },
        },

        // Background semantic colors
        background: {
          primary: '#0f1115',
          secondary: '#2d3139',
          tertiary: '#374151',
          elevated: '#1c1f2e',
          hover: '#4b5563',
        },
      },

      // Metallic gradients
      backgroundImage: {
        'gradient-metal': 'linear-gradient(135deg, #2d3139 0%, #3a3f4d 25%, #2d3139 50%, #252830 75%, #1a1d23 100%)',
        'gradient-steel': 'linear-gradient(180deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 50%, rgba(0, 0, 0, 0.1) 100%)',
        'gradient-primary': 'linear-gradient(135deg, #4f46e5 0%, #3730a3 100%)',
        'gradient-header': 'linear-gradient(135deg, #2d3139 0%, #1e1b4b 100%)',
      },

      // Metallic box shadows
      boxShadow: {
        'inset-metal': 'inset 0 1px 2px rgba(0, 0, 0, 0.5), inset 0 -1px 0 rgba(255, 255, 255, 0.05)',
        'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.4)',
        'DEFAULT': '0 1px 3px 0 rgba(0, 0, 0, 0.5), 0 1px 2px 0 rgba(0, 0, 0, 0.3)',
        'md': '0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3)',
        'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -2px rgba(0, 0, 0, 0.3)',
        'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.7), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
        'metal': '0 4px 6px -1px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05), inset 0 -1px 0 rgba(0, 0, 0, 0.3)',
      },

      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
        mono: ['JetBrains Mono', 'Monaco', 'Menlo', 'monospace'],
      },

      // Premium animations
      animation: {
        'fade-in': 'fade-in 0.4s ease-out',
        'slide-in': 'slide-in 0.4s ease-out',
        'slide-up': 'slide-up 0.3s ease-out',
        'scale-in': 'scale-in 0.2s ease-out',
        'shine': 'shine 8s linear infinite',
      },

      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(5px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in': {
          '0%': { opacity: '0', transform: 'translateX(-10px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(0.98)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        'shine': {
          '0%, 100%': { left: '-100%' },
          '50%': { left: '200%' },
        },
      },

      // Transitions
      transitionDuration: {
        fast: '150ms',
        base: '200ms',
        slow: '400ms',
      },
    },
  },
  plugins: [],
} satisfies Config
