import type { Meta, StoryObj } from '@storybook/vue3'
import RAGInterpretation from './RAGInterpretation.vue'
import { diagnosisMocks } from '~/mocks/diagnosis'

const meta = {
  title: 'Diagnosis/RAGInterpretation',
  component: RAGInterpretation,
  tags: ['autodocs'],
} satisfies Meta<typeof RAGInterpretation>

export default meta
type Story = StoryObj<typeof meta>

export const Warning: Story = {
  args: {
    interpretation: diagnosisMocks.standard.rag_interpretation,
    loading: false,
  },
}

export const Critical: Story = {
  args: {
    interpretation: diagnosisMocks.critical.rag_interpretation,
    loading: false,
  },
}

export const Normal: Story = {
  args: {
    interpretation: diagnosisMocks.normal.rag_interpretation,
    loading: false,
  },
}

export const Loading: Story = {
  args: {
    interpretation: null,
    loading: true,
  },
}

export const Error: Story = {
  args: {
    interpretation: null,
    loading: false,
    error: 'Ошибка генерации ответа RAG',
  },
}
