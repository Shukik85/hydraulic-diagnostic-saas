import { ref } from 'vue'

export const useReports = () => {
  const reports = ref([])
  const loading = ref(false)
  const error = ref(null)

  const config = useRuntimeConfig()
  const apiBase = config.public.apiBase || 'http://localhost:8000/api'

  /**
   * Получение списка отчетов для выбранной системы
   * @param {number|string} systemId - ID системы
   */
  const fetchReports = async (systemId) => {
    if (!systemId) {
      error.value = 'Не указан ID системы'
      return
    }

    loading.value = true
    error.value = null

    try {
      const token = localStorage.getItem('token')
      const headers = {
        'Content-Type': 'application/json',
      }

      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(`${apiBase}/systems/${systemId}/reports/`, {
        method: 'GET',
        headers,
      })

      if (!response.ok) {
        throw new Error(`Ошибка загрузки отчетов: ${response.status}`)
      }

      const data = await response.json()
      reports.value = data
    } catch (err) {
      console.error('Ошибка при загрузке отчетов:', err)
      error.value = err.message || 'Не удалось загрузить отчеты'
      reports.value = []
    } finally {
      loading.value = false
    }
  }

  /**
   * Создание нового отчета для системы
   * @param {number|string} systemId - ID системы
   * @param {Object} reportData - Данные отчета (title, description)
   * @returns {Promise<boolean>} - Успешность создания
   */
  const createReport = async (systemId, reportData) => {
    if (!systemId) {
      error.value = 'Не указан ID системы'
      return false
    }

    if (!reportData.title || !reportData.title.trim()) {
      error.value = 'Название отчета обязательно'
      return false
    }

    loading.value = true
    error.value = null

    try {
      const token = localStorage.getItem('token')
      const headers = {
        'Content-Type': 'application/json',
      }

      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(`${apiBase}/systems/${systemId}/reports/`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          title: reportData.title.trim(),
          description: reportData.description || '',
          system_id: systemId,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Ошибка создания отчета: ${response.status}`)
      }

      const newReport = await response.json()
      // Добавляем новый отчет в начало списка
      reports.value = [newReport, ...reports.value]
      return true
    } catch (err) {
      console.error('Ошибка при создании отчета:', err)
      error.value = err.message || 'Не удалось создать отчет'
      return false
    } finally {
      loading.value = false
    }
  }

  return {
    reports,
    loading,
    error,
    fetchReports,
    createReport,
  }
}
