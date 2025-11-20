import { mount } from '@vue/test-utils'
import { describe, it, expect } from 'vitest'
import RAGInterpretation from '../RAGInterpretation.vue'
import type { RAGInterpretationResponse } from '~/types/rag'

describe('RAGInterpretation', () => {
  const mockInterpretation: RAGInterpretationResponse = {
    reasoning: 'Test reasoning process',
    summary: 'Test summary of diagnosis',
    analysis: 'Test detailed analysis',
    recommendations: ['Recommendation 1', 'Recommendation 2'],
    confidence: 0.85,
    knowledgeUsed: [
      {
        source: 'Manual Chapter 5',
        relevance: 0.9,
        excerpt: 'Test excerpt'
      }
    ],
    metadata: {
      model: 'test-model',
      processingTime: 100,
      tokensUsed: 50
    }
  }

  it('renders interpretation correctly', () => {
    const wrapper = mount(RAGInterpretation, {
      props: {
        interpretation: mockInterpretation,
        loading: false
      }
    })

    expect(wrapper.text()).toContain('Test summary')
    expect(wrapper.text()).toContain('Test detailed analysis')
  })

  it('shows loading state', () => {
    const wrapper = mount(RAGInterpretation, {
      props: {
        interpretation: null,
        loading: true
      }
    })

    expect(wrapper.text()).toContain('Загрузка')
  })
})
