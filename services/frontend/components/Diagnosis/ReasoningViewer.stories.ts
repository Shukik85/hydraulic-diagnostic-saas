import ReasoningViewer from './ReasoningViewer.vue'
import { diagnosisMocks } from '~/mocks/diagnosis-result.mock'

export default {
  title: 'Diagnosis/ReasoningViewer',
  component: ReasoningViewer,
  argTypes: {
    reasoning: { control: 'object' },
    metadata: { control: 'object' }
  }
}

const Template = (args) => ({
  components: { ReasoningViewer },
  setup: () => ({ args }),
  template: '<ReasoningViewer v-bind="args"/>'
})

export const Default = Template.bind({})
Default.args = {
  reasoning: diagnosisMocks.standard.rag_interpretation.reasoning,
  metadata: diagnosisMocks.standard.rag_interpretation.metadata
}
