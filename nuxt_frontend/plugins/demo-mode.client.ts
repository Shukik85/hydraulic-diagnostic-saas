export default defineNuxtPlugin(() => {
  const demo = useState('demoMode', () => true)
  return { provide: { demoMode: demo } }
})