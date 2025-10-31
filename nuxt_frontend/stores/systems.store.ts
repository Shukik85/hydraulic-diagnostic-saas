// Systems store with safe API integration
import type { HydraulicSystem } from '~/types/api'

export const useSystemsStore = defineStore('systems', () => {
  const systems = ref<HydraulicSystem[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  
  // Safe API access
  const api = () => {
    try {
      return useApi().api
    } catch {
      return null
    }
  }
  
  const fetchSystems = async () => {
    loading.value = true
    error.value = null
    
    try {
      const $api = api()
      if ($api) {
        systems.value = await ($api as any)<HydraulicSystem[]>('/systems/')
      }
    } catch (err: any) {
      error.value = err?.message || 'Ошибка загрузки систем'
    } finally {
      loading.value = false
    }
  }
  
  const getSystemById = (id: number) => {
    return systems.value.find(s => s.id === id) || null
  }
  
  const addSystem = async (systemData: Partial<HydraulicSystem>) => {
    const $api = api()
    if ($api) {
      const newSystem = await ($api as any)<HydraulicSystem>('/systems/', {
        method: 'POST',
        body: systemData
      })
      systems.value.push(newSystem)
      return newSystem
    }
    return null
  }
  
  return {
    systems: readonly(systems),
    loading: readonly(loading),
    error: readonly(error),
    fetchSystems,
    getSystemById,
    addSystem
  }
})