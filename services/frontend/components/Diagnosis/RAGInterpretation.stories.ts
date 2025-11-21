import { Meta, StoryObj } from '@storybook/vue3'
import RAGInterpretation from './RAGInterpretation.vue'
import { diagnosisMocks } from '~/mocks/diagnosis'

/** @type {Meta<typeof RAGInterpretation>} */
const meta = {
  title: 'Diagnosis/RAGInterpretation',
  component: RAGInterpretation,
  tags: ['autodocs'],
}

export default meta

/** @type {StoryObj<typeof RAGInterpretation>} */
export const Warning = {
  args: {
    interpretation: diagnosisMocks.standard.rag_interpretation,
    loading: false,
  },
}

/** @type {StoryObj<typeof RAGInterpretation>} */
export const Critical = {
  args: {
    interpretation: diagnosisMocks.critical.rag_interpretation,
    loading: false,
  },
}

/** @type {StoryObj<typeof RAGInterpretation>} */
export const Normal = {
  args: {
    interpretation: diagnosisMocks.normal.rag_interpretation,
    loading: false,
  },
}

/** @type {StoryObj<typeof RAGInterpretation>} */
export const Loading = {
  args: {
    interpretation: null,
    loading: true,
  },
}

/** @type {StoryObj<typeof RAGInterpretation>} */
export const Error = {
  args: {
    interpretation: null,
    loading: false,
    error: 'Ошибка генерации ответа RAG',
  },
}
