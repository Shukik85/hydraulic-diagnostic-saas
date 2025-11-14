import { mount } from '@vue/test-utils'
import ReasoningViewer from '../ReasoningViewer.vue'
import { mockDiagnosisResult } from '~/mocks/diagnosis-result.mock'

describe('ReasoningViewer.vue', () => {
  it('renders step-by-step reasoning', () => {
    const rag = mockDiagnosisResult.rag_interpretation!
    const wrapper = mount(ReasoningViewer, {
      props: {
        reasoning: rag.reasoning,
        metadata: rag.metadata
      }
    })
    expect(wrapper.html()).toMatch(/Анализ аномалии давления/)
    expect(wrapper.findAll('.step-item').length).toBe(rag.reasoning.length)
  })

  it('highlights keywords in step text', () => {
    const reasoned = [
      { step: 1, title: 'Step', description: 'Критический риск отказа', evidence: [], conclusion: 'важно ожидать немедленно' }
    ]
    const wrapper = mount(ReasoningViewer, { props: { reasoning: reasoned } })
    expect(wrapper.html()).toMatch(/<mark>критический<\/mark>/i)
  })

  it('copies to clipboard', async () => {
    Object.assign(navigator, { clipboard: { writeText: jest.fn() } })
    const wrapper = mount(ReasoningViewer, { props: { reasoning: mockDiagnosisResult.rag_interpretation!.reasoning } })
    await wrapper.find('.actions button').trigger('click')
    expect(navigator.clipboard.writeText).toHaveBeenCalled()
  })
})
