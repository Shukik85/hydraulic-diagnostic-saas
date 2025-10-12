import axios from 'axios'

const API_URL = 'http://127.0.0.1:8000/api'

export const authService = {
  async login(credentials) {
    try {
      console.log('Sending login:', credentials)
      
      // ИСПРАВЛЕНИЕ: отправляем email вместо username
      const loginData = {
        email: credentials.username,  // Переименовываем поле
        password: credentials.password
      }
      
      const response = await axios.post(`${API_URL}/auth/login/`, loginData)
      console.log('Login response:', response.data)
      
      const { access, refresh, user } = response.data
      
      localStorage.setItem('access_token', access)
      localStorage.setItem('refresh_token', refresh)
      localStorage.setItem('user', JSON.stringify(user))
      
      return { access, refresh, user }
    } catch (error) {
      console.error('Login error:', error.response?.data)
      throw error
    }
  },

  async register(userData) {
    try {
      const response = await axios.post(`${API_URL}/auth/register/`, userData)
      return response.data
    } catch (error) {
      console.error('Ошибка регистрации:', error)
      throw error
    }
  },

  logout() {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token') 
    localStorage.removeItem('user')
  },

  isAuthenticated() {
    return !!localStorage.getItem('access_token')
  },

  getCurrentUser() {
    const user = localStorage.getItem('user')
    return user ? JSON.parse(user) : null
  }
}
