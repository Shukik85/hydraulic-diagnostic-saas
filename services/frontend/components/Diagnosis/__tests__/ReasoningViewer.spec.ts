import { mount } from '@vue/test-utils'
import { describe, it, expect, vi } from 'vitest'
import ReasoningViewer from '../ReasoningViewer.vue'
import type { ReasoningStep, RAGMetadata } from '~/types/rag'

describe('ReasoningViewer', () => {
  const mockReasoning: ReasoningStep[] = [
    {
      step: 1,
      title: 'Step 1',
      content: 'Content 1',
      confidence: 0.9
    }
  ]

  const mockMetadata: RAGMetadata = {
    model: 'test-model',
    processingTime: 100,
    tokensUsed: 50
  }

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
        metadata: mockMetadata
      }
    })

    expect(wrapper.text()).toContain('Step 1')
    expect(wrapper.text()).toContain('Content 1')
  })
})
