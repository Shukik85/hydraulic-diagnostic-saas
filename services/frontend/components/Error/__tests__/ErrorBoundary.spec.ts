import { mount } from '@vue/test-utils'
import ErrorBoundary from '../ErrorBoundary.vue'
import ErrorFallback from '../ErrorFallback.vue'
describe('ErrorBoundary.vue', () => {
  it('renders slot when no error', () => {
    const wrapper = mount(ErrorBoundary, {
      slots: { default: '<div>Success</div>' }
    })
    expect(wrapper.html()).toContain('Success')
  })
  it('renders fallback on error', async () => {
    const ErrComp = { template: '<div>{{ err.nope }}</div>' }
    const wrapper = mount(ErrorBoundary, {
      slots: { default: ErrComp, fallback: ErrorFallback }
    })
    expect(wrapper.findComponent(ErrorFallback).exists()).toBe(true)
  })
})
