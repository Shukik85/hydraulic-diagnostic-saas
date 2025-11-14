import { mount } from '@vue/test-utils'
import LoadingSpinner from '../LoadingSpinner.vue'
describe('LoadingSpinner.vue', () => {
  it('renders with specified size and variant', () => {
    const wrapper = mount(LoadingSpinner, { props: { size: 'lg', variant: 'success', text: 'Загрузка...' } })
    expect(wrapper.find('.loading-spinner').exists()).toBe(true)
    expect(wrapper.text()).toContain('Загрузка')
    expect(wrapper.classes()).toContain('spinner-lg')
    expect(wrapper.find('.spinner-success').exists()).toBe(true)
  })
  it('renders icon if specified', () => {
    const wrapper = mount(LoadingSpinner, { props: { icon: 'lucide:check', text: 'with icon' } })
    expect(wrapper.find('.spinner-icon').exists()).toBe(true)
  })
})
