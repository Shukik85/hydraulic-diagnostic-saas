import type { Meta as MetaType, StoryObj as StoryObjType } from '@storybook/vue3'
import DiagnosisProgress from './DiagnosisProgress.vue'

interface Stage {
  id: string
  name: string
  status: 'pending' | 'active' | 'complete' | 'error'
  progress?: number
  duration?: string
}

const meta: MetaType<typeof DiagnosisProgress> = {
  title: 'Loading/DiagnosisProgress',
  component: DiagnosisProgress,
  tags: ['autodocs'],
}

export default meta
type Story = StoryObjType<typeof DiagnosisProgress>

const baseStages: Stage[] = [
  { id: '1', name: 'Загрузка данных', status: 'complete' as const, duration: '0.8 сек.' },
  { id: '2', name: 'Анализ ML', status: 'active' as const, progress: 45 },
  { id: '3', name: 'Генерация отчета', status: 'pending' as const },
]

export const Active: Story = {
  args: {
    stages: baseStages,
    eta: '4 сек.',
  },
}

export const AllComplete: Story = {
  args: {
    stages: baseStages.map(s => ({ ...s, status: 'complete' as const, progress: undefined })),
    eta: '0',
  },
}
