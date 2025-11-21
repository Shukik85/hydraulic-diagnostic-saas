import type { Meta, StoryObj } from '@storybook/vue3'
import ReasoningViewer from './ReasoningViewer.vue'
import { diagnosisMocks } from '~/mocks/diagnosis'

const meta = {
  title: 'Diagnosis/ReasoningViewer',
  component: ReasoningViewer,
  tags: ['autodocs'],
} as Meta<typeof ReasoningViewer>

export default meta
type Story = StoryObj<typeof ReasoningViewer>

export const Default: Story = {
  args: {
    reasoning: diagnosisMocks.standard.rag_interpretation?.reasoning ?? [],
    metadata: diagnosisMocks.standard.rag_interpretation?.metadata ?? {
      model: 'mock',
      processingTime: 100,
      tokensUsed: 50
    },
  },
}
