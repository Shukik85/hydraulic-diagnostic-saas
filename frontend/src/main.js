import { createApp } from 'vue'
import App from './App.vue'
<<<<<<< HEAD
import router from './router'
import { createPinia } from 'pinia'
import MainLayout from './layouts/MainLayout.vue'
import websocketService, { useWebSocket } from './services/websocketService'
=======
import router from './router'  // Добавь эту строку
import './style.css'
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f

// Создание приложения
const app = createApp(App)
<<<<<<< HEAD

// Подключение Pinia для state management
const pinia = createPinia()
app.use(pinia)

// Подключение роутера
app.use(router)

// Регистрация глобальных компонентов
app.component('MainLayout', MainLayout)

// Глобальные свойства
app.config.globalProperties.$websocket = websocketService
app.provide('websocket', websocketService)

// Глобальные стили и утилиты
import './styles/main.css'

// Инициализация WebSocket при запуске приложения
websocketService.requestNotificationPermission()

// Автоматическое подключение WebSocket если пользователь аутентифицирован
const token = localStorage.getItem('token')
if (token) {
  websocketService.connect(token)
}

// Обработка роутинга для WebSocket подключения
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  
  if (to.meta.requiresAuth && token && !websocketService.getConnectionStatus().isConnected) {
    websocketService.connect(token)
  }
  
  next()
})

// Монтирование приложения
=======
app.use(router)  // Добавь эту строку
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
app.mount('#app')

// Обработка ошибок Vue
app.config.errorHandler = (err, vm, info) => {
  console.error('Vue error:', err, info)
  
  // Отправка ошибки в систему мониторинга (например, Sentry)
  if (process.env.NODE_ENV === 'production') {
    // Sentry.captureException(err)
  }
}

// Регистрация Service Worker для PWA (опционально)
if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
  navigator.serviceWorker.register('/sw.js')
    .then(registration => {
      console.log('SW registered: ', registration)
    })
    .catch(registrationError => {
      console.log('SW registration failed: ', registrationError)
    })
}

// Обработка изменения состояния сети
window.addEventListener('online', () => {
  console.log('Соединение с интернетом восстановлено')
  if (token && !websocketService.getConnectionStatus().isConnected) {
    websocketService.connect(token)
  }
})

window.addEventListener('offline', () => {
  console.log('Соединение с интернетом потеряно')
})

// Экспорт приложения для тестов
export default app
