import type { Meta, StoryObj } from '@storybook/vue3'
import ErrorFallback from './ErrorFallback.vue'

const meta: Meta<typeof ErrorFallback> = {
  title: 'Error/ErrorFallback',
  component: ErrorFallback,
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ErrorFallback>

export const Default: Story = {
  args: {
    error: new Error('Тестовая ошибка'),
  },
}

export const NetworkError: Story = {
  args: {
    error: new Error('Network request failed'),
  },
}
