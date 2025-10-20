// composables/useSystems.js
import { ref } from 'vue'

/**
 * Composable для работы со списком гидравлических систем
 * Обеспечивает получение, фильтрацию и управление данными систем
 */
export const useSystems = () => {
  // Реактивные состояния
  const systems = ref([])
  const loading = ref(false)
  const error = ref(null)

  // Конфигурация API
  const config = useRuntimeConfig()
  const API_BASE_URL = config.public.apiBaseUrl || 'http://localhost:8000/api'

  /**
   * Получение списка всех систем
   */
  const getSystems = async () => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch(`${API_BASE_URL}/systems/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      systems.value = response
      return response
    } catch (err) {
      console.error('Ошибка при загрузке систем:', err)
      error.value = err.message || 'Не удалось загрузить список систем'
      systems.value = []
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Получение информации об одной системе по ID
   * @param {number|string} systemId - ID системы
   */
  const getSystemById = async (systemId) => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch(`${API_BASE_URL}/systems/${systemId}/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      return response
    } catch (err) {
      console.error(`Ошибка при загрузке системы ${systemId}:`, err)
      error.value = err.message || 'Не удалось загрузить данные системы'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Фильтрация систем по статусу
   * @param {string} status - Статус для фильтрации (active, warning, error, offline)
   */
  const filterByStatus = (status) => {
    if (!status) return systems.value
    return systems.value.filter(system => system.status === status)
  }

  /**
   * Поиск систем по названию или местоположению
   * @param {string} query - Поисковый запрос
   */
  const searchSystems = (query) => {
    if (!query) return systems.value

    const searchLower = query.toLowerCase()
    return systems.value.filter(system =>
      system.name.toLowerCase().includes(searchLower) ||
      system.location?.toLowerCase().includes(searchLower)
    )
  }

  /**
   * Получение статистики по системам
   */
  const getSystemsStats = () => {
    const stats = {
      total: systems.value.length,
      active: 0,
      warning: 0,
      error: 0,
      offline: 0
    }

    systems.value.forEach(system => {
      if (stats.hasOwnProperty(system.status)) {
        stats[system.status]++
      }
    })

    return stats
  }

  // Возвращаем API composable
  return {
    // Состояния
    systems,
    loading,
    error,

    // Методы
    getSystems,
    getSystemById,
    filterByStatus,
    searchSystems,
    getSystemsStats,
  }
}
