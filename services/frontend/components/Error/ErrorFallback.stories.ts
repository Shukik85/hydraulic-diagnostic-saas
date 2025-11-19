import ErrorFallback from '~/components/Error/ErrorFallback.vue'
export default {
  title: 'Diagnosis/ErrorFallback',
  component: ErrorFallback,
  argTypes: { error: { control: 'object' } }
}
const Template = (args) => ({
  components: { ErrorFallback },
  setup: () => ({ args }),
  template: '<ErrorFallback v-bind="args"/>'
})
export const Default = Template.bind({})
Default.args = { error: new Error('Тестовая ошибка') }
