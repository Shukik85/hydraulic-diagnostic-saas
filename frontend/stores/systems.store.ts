// Systems store with safe API integration
import type { HydraulicSystem } from '~/types/api'

export const useSystemsStore = defineStore('systems', () => {
  const systems = ref<HydraulicSystem[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  const api = () => { try { return useApi().api } catch { return null } }

  const fetchSystems = async () => {
    loading.value = true; error.value = null
    try {
      const $api = api()
      if ($api) {
        const res = await ($api as any)('/systems/')
        systems.value = Array.isArray(res) ? res as HydraulicSystem[] : []
      }
    } catch (err: any) { error.value = err?.message || 'Ошибка загрузки систем' }
    finally { loading.value = false }
  }

  const getSystemById = (id: number) => systems.value.find(s => s.id === id) || null

  const addSystem = async (systemData: Partial<HydraulicSystem>) => {
    const $api = api()
    if ($api) {
      const newSystem = await ($api as any)('/systems/', { method: 'POST', body: systemData })
      systems.value.push(newSystem as HydraulicSystem)
      return newSystem as HydraulicSystem
    }
    return null
  }

  return { systems: readonly(systems), loading: readonly(loading), error: readonly(error), fetchSystems, getSystemById, addSystem }
})
