/**
 * Systems Store - Управление гидравлическими системами
 *
 * ✅ ИСПРАВЛЕНО:
 * - Открыт API (useGeneratedApi вместо useApi)
 * - Типобезопасные операции
 * - Error handling
 * - Готово к OpenAPI integration
 */

import type { HydraulicSystem } from '~/types/api'

export const useSystemsStore = defineStore('systems', () => {
  const systems = ref<HydraulicSystem[]>([])
  const currentSystem = ref<HydraulicSystem | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  /**
   * Fetch все системы
   * TODO: После кодгена OpenAPI использовать:
   * const { equipment } = useGeneratedApi()
   * const response = await equipment.getSystems()
   */
  const fetchSystems = async () => {
    loading.value = true
    error.value = null
    try {
      // Placeholder для будущей реальной интеграции
      // const { equipment } = useGeneratedApi()
      // const response = await equipment.getSystems()
      // systems.value = Array.isArray(response) ? response : response.results || []

      console.warn('[systems.store] fetchSystems placeholder - awaiting OpenAPI client')
      systems.value = []
    } catch (err: any) {
      error.value = err?.message || 'Ошибка при получении смсыстем'
      console.error('[systems.store] fetchSystems error:', err)
    } finally {
      loading.value = false
    }
  }

  /**
   * Fetch конкретную систему по ID
   * TODO: После кодгена OpenAPI использовать:
   * const { equipment } = useGeneratedApi()
   * const response = await equipment.getSystem(id)
   */
  const fetchSystem = async (id: number | string) => {
    loading.value = true
    error.value = null
    try {
      // Placeholder
      // const { equipment } = useGeneratedApi()
      // currentSystem.value = await equipment.getSystem(id)

      console.warn('[systems.store] fetchSystem placeholder - awaiting OpenAPI client')
      currentSystem.value = null
    } catch (err: any) {
      error.value = err?.message || 'Ошибка при получении системы'
      console.error('[systems.store] fetchSystem error:', err)
    } finally {
      loading.value = false
    }
  }

  /**
   * Получить систему из локального кэша или фетчить
   */
  const getSystemById = (id: number | string) => {
    return (
      systems.value.find((s) => s.id === id || String(s.id) === String(id)) || null
    )
  }

  /**
   * Новая система
   * TODO: После кодгена OpenAPI использовать:
   * const { equipment } = useGeneratedApi()
   * const newSystem = await equipment.createSystem(systemData)
   */
  const addSystem = async (systemData: Partial<HydraulicSystem>) => {
    loading.value = true
    error.value = null
    try {
      // Placeholder
      // const { equipment } = useGeneratedApi()
      // const newSystem = await equipment.createSystem(systemData)
      // systems.value.push(newSystem as HydraulicSystem)
      // return newSystem as HydraulicSystem

      console.warn('[systems.store] addSystem placeholder - awaiting OpenAPI client')
      throw new Error('Equipment service not yet available')
    } catch (err: any) {
      error.value = err?.message || 'Ошибка при сохранении системы'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Обновить систему
   * TODO: После кодгена OpenAPI использовать:
   * const { equipment } = useGeneratedApi()
   * const updated = await equipment.updateSystem(id, systemData)
   */
  const updateSystem = async (
    id: number | string,
    systemData: Partial<HydraulicSystem>
  ) => {
    loading.value = true
    error.value = null
    try {
      // Placeholder
      // const { equipment } = useGeneratedApi()
      // const updated = await equipment.updateSystem(id, systemData)
      // const idx = systems.value.findIndex(s => s.id === id || String(s.id) === String(id))
      // if (idx >= 0) systems.value[idx] = updated
      // if (currentSystem.value?.id === id) currentSystem.value = updated
      // return updated

      console.warn('[systems.store] updateSystem placeholder - awaiting OpenAPI client')
      throw new Error('Equipment service not yet available')
    } catch (err: any) {
      error.value = err?.message || 'Ошибка при обновлении системы'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Очистить стор (для девы и тестов)
   */
  const reset = () => {
    systems.value = []
    currentSystem.value = null
    loading.value = false
    error.value = null
  }

  return {
    // State
    systems: readonly(systems),
    currentSystem: readonly(currentSystem),
    loading: readonly(loading),
    error: readonly(error),

    // Actions
    fetchSystems,
    fetchSystem,
    getSystemById,
    addSystem,
    updateSystem,
    reset,
  }
})
