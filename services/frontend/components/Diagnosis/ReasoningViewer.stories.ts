import type { Meta as MetaType, StoryObj as StoryObjType } from '@storybook/vue3'
import ReasoningViewer from './ReasoningViewer.vue'
import { diagnosisMocks } from '~/mocks/diagnosis'

const meta: MetaType<typeof ReasoningViewer> = {
  title: 'Diagnosis/ReasoningViewer',
  component: ReasoningViewer,
  tags: ['autodocs'],
}

export default meta
type Story = StoryObjType<typeof ReasoningViewer>

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
