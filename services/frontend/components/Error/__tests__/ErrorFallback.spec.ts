import { mount } from '@vue/test-utils'
import ErrorFallback from '../ErrorFallback.vue'

describe('ErrorFallback.vue', () => {
  it('renders error title and message', () => {
    const wrapper = mount(ErrorFallback, { props: { error: new Error('Тестовая ошибка') } })
    expect(wrapper.text()).toContain('Что-то пошло не так')
    expect(wrapper.text()).toContain('Тестовая ошибка')
  })
  it('emits reset event', async () => {
    const wrapper = mount(ErrorFallback, { props: { error: new Error('Ошибка') } })
    await wrapper.find('button').trigger('click')
    expect(wrapper.emitted('reset')).toBeTruthy()
  })
})
