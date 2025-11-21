import { Meta, StoryObj } from '@storybook/vue3'
import ErrorFallback from './ErrorFallback.vue'

/** @type {Meta<typeof ErrorFallback>} */
const meta = {
  title: 'Error/ErrorFallback',
  component: ErrorFallback,
  tags: ['autodocs'],
}

export default meta

/** @type {StoryObj<typeof ErrorFallback>} */
export const Default = {
  args: {
    error: new Error('Тестовая ошибка'),
  },
}

/** @type {StoryObj<typeof ErrorFallback>} */
export const NetworkError = {
  args: {
    error: new Error('Network request failed'),
  },
}
