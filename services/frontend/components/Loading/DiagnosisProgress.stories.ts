import { Meta, StoryObj } from '@storybook/vue3'
import DiagnosisProgress from './DiagnosisProgress.vue'

interface Stage {
  id: string
  name: string
  status: 'pending' | 'active' | 'complete' | 'error'
  progress?: number
  duration?: string
}

/** @type {Meta<typeof DiagnosisProgress>} */
const meta = {
  title: 'Loading/DiagnosisProgress',
  component: DiagnosisProgress,
  tags: ['autodocs'],
}

export default meta

const baseStages: Stage[] = [
  { id: '1', name: 'Загрузка данных', status: 'complete' as const, duration: '0.8 сек.' },
  { id: '2', name: 'Анализ ML', status: 'active' as const, progress: 45 },
  { id: '3', name: 'Генерация отчета', status: 'pending' as const },
]

/** @type {StoryObj<typeof DiagnosisProgress>} */
export const Active = {
  args: {
    stages: baseStages,
    eta: '4 сек.',
  },
}

/** @type {StoryObj<typeof DiagnosisProgress>} */
export const AllComplete = {
  args: {
    stages: baseStages.map(s => ({ ...s, status: 'complete' as const, progress: undefined })),
    eta: '0',
  },
}
