import { Meta, StoryObj } from '@storybook/vue3'
import ReasoningViewer from './ReasoningViewer.vue'
import { diagnosisMocks } from '~/mocks/diagnosis'

/** @type {Meta<typeof ReasoningViewer>} */
const meta = {
  title: 'Diagnosis/ReasoningViewer',
  component: ReasoningViewer,
  tags: ['autodocs'],
}

export default meta

/** @type {StoryObj<typeof ReasoningViewer>} */
export const Default = {
  args: {
    reasoning: diagnosisMocks.standard.rag_interpretation?.reasoning ?? [],
    metadata: diagnosisMocks.standard.rag_interpretation?.metadata ?? {
      model: 'mock',
      processingTime: 100,
      tokensUsed: 50
    },
  },
}
