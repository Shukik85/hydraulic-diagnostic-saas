import apiClient from './api'

export const hydraulicSystemService = {
  // Получить список систем
  async getSystems() {
    try {
      const response = await apiClient.get('hydraulic-systems/')
      return response.data
    } catch (error) {
      console.error('Ошибка получения списка систем:', error)
      throw error
    }
  },

  // Получить детали системы
  async getSystem(id) {
    try {
      const response = await apiClient.get(`hydraulic-systems/${id}/`)
      return response.data
    } catch (error) {
      console.error(`Ошибка получения системы ${id}:`, error)
      throw error
    }
  },

  // Создать новую систему
  async createSystem(systemData) {
    try {
      const response = await apiClient.post('hydraulic-systems/', systemData)
      return response.data
    } catch (error) {
      console.error('Ошибка создания системы:', error)
      throw error
    }
  },

  // Обновить систему
  async updateSystem(id, systemData) {
    try {
      const response = await apiClient.put(`hydraulic-systems/${id}/`, systemData)
      return response.data
    } catch (error) {
      console.error(`Ошибка обновления системы ${id}:`, error)
      throw error
    }
  },

  // Удалить систему
  async deleteSystem(id) {
    try {
      await apiClient.delete(`hydraulic-systems/${id}/`)
      return true
    } catch (error) {
      console.error(`Ошибка удаления системы ${id}:`, error)
      throw error
    }
  },

  // Получить данные датчиков системы
  async getSystemSensorData(id) {
    try {
      const response = await apiClient.get(`hydraulic-systems/${id}/sensor_data/`)
      return response.data
    } catch (error) {
      console.error(`Ошибка получения данных датчиков системы ${id}:`, error)
      throw error
    }
  },

  // Получить отчеты системы
  async getSystemReports(id) {
    try {
      const response = await apiClient.get(`hydraulic-systems/${id}/reports/`)
      return response.data
    } catch (error) {
      console.error(`Ошибка получения отчетов системы ${id}:`, error)
      throw error
    }
  },

  // Запустить диагностику системы
  async diagnoseSystem(id) {
    try {
      const response = await apiClient.post(`hydraulic-systems/${id}/diagnose/`)
      return response.data
    } catch (error) {
      console.error(`Ошибка диагностики системы ${id}:`, error)
      throw error
    }
  },

  // Генерация тестовых данных
  async generateTestData() {
    try {
      const response = await apiClient.post('hydraulic-systems/generate_test_data/')
      return response.data
    } catch (error) {
      console.error('Ошибка генерации тестовых данных:', error)
      throw error
    }
  },

  // Загрузка данных датчиков
  async uploadSensorData(systemId, formData) {
    try {
      const response = await apiClient.post(
        `hydraulic-systems/${systemId}/upload_sensor_data/`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )
      return response.data
    } catch (error) {
      console.error(`Ошибка загрузки данных для системы ${systemId}:`, error)
      throw error
    }
  }
}

