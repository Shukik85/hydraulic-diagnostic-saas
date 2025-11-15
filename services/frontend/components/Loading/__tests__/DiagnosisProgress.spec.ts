import { mount } from '@vue/test-utils'
import DiagnosisProgress from '../DiagnosisProgress.vue'

describe('DiagnosisProgress.vue', () => {
  const mockStages = [
    { id: 'prepare', name: 'Подготовка', status: 'complete', duration: '0.4s' },
    { id: 'gnn', name: 'GNN', status: 'complete', duration: '1.3s' },
    { id: 'rag', name: 'RAG', status: 'active', progress: 42 },
    { id: 'report', name: 'Отчёт', status: 'pending' }
  ]

  it('renders all stages, correct statuses and progress', () => {
    const wrapper = mount(DiagnosisProgress, { props: { stages: mockStages, eta: '4.5 сек.' }})
    const stageEls = wrapper.findAll('.stage-item')
    expect(stageEls.length).toBe(mockStages.length)
    expect(wrapper.text()).toContain('RAG')
    expect(wrapper.text()).toContain('42%')
    expect(wrapper.text()).toContain('4.5 сек.')
  })

  it('updates status classes', () => {
    const wrapper = mount(DiagnosisProgress, { props: { stages: mockStages }})
    expect(wrapper.find('.stage-item.stage-active').exists()).toBe(true)
    expect(wrapper.find('.stage-item.stage-complete').exists()).toBe(true)
  })
})
