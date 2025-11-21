import { mount } from '@vue/test-utils'
import { describe, it, expect, vi } from 'vitest'
import ReasoningViewer from '../ReasoningViewer.vue'
import type { ReasoningStep } from '~/types/rag'

describe('ReasoningViewer', () => {
  const mockReasoning: ReasoningStep[] = [
    {
      title: 'Step 1',
      description: 'Content 1',
      evidence: ['Evidence point 1', 'Evidence point 2']
    }
  ]

  // Mock clipboard API для vitest
  Object.assign(navigator, {
    clipboard: {
      writeText: vi.fn(() => Promise.resolve())
    }
  })

  it('renders reasoning steps', () => {
    const wrapper = mount(ReasoningViewer, {
      props: {
        reasoning: mockReasoning,
        metadata: {
          model: 'test-model',
          processingTime: 100,
          tokensUsed: 50
        }
      }
    })

    expect(wrapper.text()).toContain('Step 1')
    expect(wrapper.text()).toContain('Content 1')
  })
})
