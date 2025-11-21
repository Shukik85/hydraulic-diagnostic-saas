import { mount } from '@vue/test-utils'
import { describe, it, expect } from 'vitest'
import DiagnosisProgress from '../DiagnosisProgress.vue'

interface Stage {
  id: string
  name: string
  status: 'pending' | 'active' | 'complete' | 'error'
  progress?: number
  duration?: string
}

describe('DiagnosisProgress', () => {
  const mockStages: Stage[] = [
    { id: '1', name: 'Data Loading', status: 'complete' as const, duration: '1.2 сек.' },
    { id: '2', name: 'ML Analysis', status: 'active' as const, progress: 65 },
    { id: '3', name: 'Report Generation', status: 'pending' as const }
  ]

  it('renders all stages', () => {
    const wrapper = mount(DiagnosisProgress, { props: { stages: mockStages, eta: '4.5 сек.' }})
    
    expect(wrapper.text()).toContain('Data Loading')
    expect(wrapper.text()).toContain('ML Analysis')
    expect(wrapper.text()).toContain('Report Generation')
  })

  it('renders without eta', () => {
    const wrapper = mount(DiagnosisProgress, { props: { stages: mockStages }})
    
    expect(wrapper.find('.stages-container').exists()).toBe(true)
  })
})
