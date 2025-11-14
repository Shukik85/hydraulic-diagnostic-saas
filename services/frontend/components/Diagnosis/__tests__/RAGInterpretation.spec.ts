import { mount } from '@vue/test-utils'
import RAGInterpretation from '../RAGInterpretation.vue'
import { mockDiagnosisResult } from '~/mocks/diagnosis-result.mock'

describe('RAGInterpretation.vue', () => {
  it('renders summary and recommendations from data', () => {
    const rag = mockDiagnosisResult.rag_interpretation
    const wrapper = mount(RAGInterpretation, {
      props: {
        interpretation: rag,
        loading: false,
        error: null
      }
    })
    expect(wrapper.text()).toContain('утечка давления')
    expect(wrapper.text()).toContain('Проверить уплотнения насоса')
    expect(wrapper.findAll('.recommendation-item').length).toBe(rag!.recommendations.length)
  })

  it('renders loading state', () => {
    const wrapper = mount(RAGInterpretation, {
      props: {
        loading: true
      }
    })
    expect(wrapper.text()).toContain('Анализ результатов диагностики')
  })

  it('renders error state', () => {
    const wrapper = mount(RAGInterpretation, {
      props: {
        error: 'Ошибка генерации интерпретации'
      }
    })
    expect(wrapper.text()).toContain('Интерпретация недоступна')
    expect(wrapper.text()).toContain('Ошибка генерации интерпретации')
  })

  it('renders empty state when no interpretation', () => {
    const wrapper = mount(RAGInterpretation)
    expect(wrapper.text()).toContain('Нет данных для интерпретации')
  })
})
