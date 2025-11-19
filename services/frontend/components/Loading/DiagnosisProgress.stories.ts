import DiagnosisProgress from './DiagnosisProgress.vue'

const baseStages = [
  { id: 'prepare', name: 'Подготовка', status: 'complete', duration: '0.7с' },
  { id: 'gnn', name: 'GNN', status: 'complete', duration: '1.3с' },
  { id: 'rag', name: 'RAG', status: 'active', progress: 63 },
  { id: 'report', name: 'Отчет', status: 'pending' }
]

export default {
  title: 'Diagnosis/DiagnosisProgress',
  component: DiagnosisProgress,
  argTypes: {
    stages: { control: 'object' },
    eta: { control: 'text' }
  }
}

const Template = (args) => ({
  components: { DiagnosisProgress },
  setup: () => ({ args }),
  template: '<DiagnosisProgress v-bind="args"/>'
})

export const Active = Template.bind({})
Active.args = { stages: baseStages, eta: '4 сек.' }

export const AllComplete = Template.bind({})
AllComplete.args = { stages: baseStages.map(s => ({ ...s, status: 'complete', progress: undefined })), eta: '0' }
