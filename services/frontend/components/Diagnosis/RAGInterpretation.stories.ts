import RAGInterpretation from './RAGInterpretation.vue'
import { diagnosisMocks } from '~/mocks/diagnosis-result.mock'

export default {
  title: 'Diagnosis/RAGInterpretation',
  component: RAGInterpretation,
  argTypes: {
    interpretation: { control: 'object' },
    loading: { control: 'boolean' },
    error: { control: 'text' }
  }
}

const Template = (args) => ({
  components: { RAGInterpretation },
  setup: () => ({ args }),
  template: '<RAGInterpretation v-bind="args"/>'
})

export const Warning = Template.bind({})
Warning.args = { interpretation: diagnosisMocks.standard.rag_interpretation }

export const Critical = Template.bind({})
Critical.args = { interpretation: diagnosisMocks.critical.rag_interpretation }

export const Normal = Template.bind({})
Normal.args = { interpretation: diagnosisMocks.normal.rag_interpretation }

export const Loading = Template.bind({})
Loading.args = { loading: true }

export const Error = Template.bind({})
Error.args = { error: 'Ошибка генерации ответа RAG' }
