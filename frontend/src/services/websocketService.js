/**
 * WebSocket сервис для real-time обновлений
 */
class WebSocketService {
  constructor() {
    this.ws = null
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
    this.reconnectInterval = 5000 // 5 секунд
    this.listeners = new Map()
    this.isConnected = false
    this.heartbeatInterval = null
    this.heartbeatTimeout = 30000 // 30 секунд
    
    // Автоматическое переподключение
    this.shouldReconnect = true
    
    // Очередь сообщений для отправки когда соединение восстановится
    this.messageQueue = []
  }

  /**
   * Подключение к WebSocket серверу
   */
  connect(token = null) {
    try {
      // Определение URL WebSocket сервера
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.host
      const wsUrl = `${protocol}//${host}/ws/diagnostics/`
      
      console.log('🔌 Подключение к WebSocket:', wsUrl)
      
      // Создание WebSocket соединения
      this.ws = new WebSocket(wsUrl)
      
      // Обработчики событий
      this.ws.onopen = this.onOpen.bind(this)
      this.ws.onmessage = this.onMessage.bind(this)
      this.ws.onclose = this.onClose.bind(this)
      this.ws.onerror = this.onError.bind(this)
      
    } catch (error) {
      console.error('❌ Ошибка подключения WebSocket:', error)
      this.handleReconnect()
    }
  }

  /**
   * Обработчик открытия соединения
   */
  onOpen(event) {
    console.log('✅ WebSocket соединение установлено')
    this.isConnected = true
    this.reconnectAttempts = 0
    
    // Аутентификация если есть токен
    const token = localStorage.getItem('token')
    if (token) {
      this.send({
        type: 'auth',
        token: token
      })
    }
    
    // Запуск heartbeat
    this.startHeartbeat()
    
    // Отправка накопленных сообщений
    this.flushMessageQueue()
    
    // Уведомление слушателей о подключении
    this.emit('connected', { status: 'connected' })
  }

  /**
   * Обработчик получения сообщений
   */
  onMessage(event) {
    try {
      const data = JSON.parse(event.data)
      console.log('📨 Получено WebSocket сообщение:', data)
      
      // Обработка системных сообщений
      switch (data.type) {
        case 'auth_success':
          console.log('🔐 Аутентификация успешна')
          this.subscribeToUpdates()
          break
          
        case 'auth_failed':
          console.error('🚫 Ошибка аутентификации WebSocket')
          this.emit('auth_failed', data)
          break
          
        case 'pong':
          // Ответ на ping - соединение живо
          break
          
        case 'sensor_data':
          this.handleSensorData(data)
          break
          
        case 'critical_alert':
          this.handleCriticalAlert(data)
          break
          
        case 'diagnostic_result':
          this.handleDiagnosticResult(data)
          break
          
        case 'system_status_change':
          this.handleSystemStatusChange(data)
          break
          
        default:
          // Передача сообщения слушателям
          this.emit(data.type, data)
      }
    } catch (error) {
      console.error('❌ Ошибка парсинга WebSocket сообщения:', error)
    }
  }

  /**
   * Обработчик закрытия соединения
   */
  onClose(event) {
    console.log('🔌 WebSocket соединение закрыто:', event.code, event.reason)
    this.isConnected = false
    this.stopHeartbeat()
    
    // Уведомление слушателей о разъединении
    this.emit('disconnected', { 
      code: event.code, 
      reason: event.reason 
    })
    
    // Автоматическое переподключение если нужно
    if (this.shouldReconnect && event.code !== 1000) {
      this.handleReconnect()
    }
  }

  /**
   * Обработчик ошибок
   */
  onError(error) {
    console.error('❌ WebSocket ошибка:', error)
    this.emit('error', { error })
  }

  /**
   * Отправка сообщения
   */
  send(message) {
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      try {
        const jsonMessage = JSON.stringify(message)
        this.ws.send(jsonMessage)
        console.log('📤 Отправлено WebSocket сообщение:', message)
      } catch (error) {
        console.error('❌ Ошибка отправки сообщения:', error)
      }
    } else {
      // Добавление в очередь если не подключено
      console.log('📋 Сообщение добавлено в очередь:', message)
      this.messageQueue.push(message)
    }
  }

  /**
   * Отправка накопленных сообщений
   */
  flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()
      this.send(message)
    }
  }

  /**
   * Подписка на обновления после аутентификации
   */
  subscribeToUpdates() {
    this.send({
      type: 'subscribe',
      channels: [
        'sensor_data',
        'critical_alerts', 
        'diagnostic_results',
        'system_status'
      ]
    })
  }

  /**
   * Обработка данных датчиков
   */
  handleSensorData(data) {
    this.emit('sensor_data_update', {
      systemId: data.system_id,
      sensorType: data.sensor_type,
      value: data.value,
      unit: data.unit,
      timestamp: data.timestamp,
      isCritical: data.is_critical,
      warningMessage: data.warning_message
    })
  }

  /**
   * Обработка критических предупреждений
   */
  handleCriticalAlert(data) {
    console.warn('🚨 Критическое предупреждение:', data)
    
    this.emit('critical_alert', {
      systemId: data.system_id,
      systemName: data.system_name,
      alertType: data.alert_type,
      message: data.message,
      severity: data.severity,
      timestamp: data.timestamp,
      recommendedActions: data.recommended_actions || []
    })
    
    // Показ браузерного уведомления если разрешено
    this.showBrowserNotification(
      'Критическое предупреждение',
      `${data.system_name}: ${data.message}`,
      'warning'
    )
  }

  /**
   * Обработка результатов диагностики
   */
  handleDiagnosticResult(data) {
    this.emit('diagnostic_completed', {
      systemId: data.system_id,
      reportId: data.report_id,
      result: data.result,
      timestamp: data.timestamp
    })
    
    // Уведомление о завершении диагностики
    this.showBrowserNotification(
      'Диагностика завершена',
      `Анализ системы "${data.system_name}" выполнен`,
      'info'
    )
  }

  /**
   * Обработка изменения статуса системы
   */
  handleSystemStatusChange(data) {
    this.emit('system_status_changed', {
      systemId: data.system_id,
      oldStatus: data.old_status,
      newStatus: data.new_status,
      timestamp: data.timestamp
    })
  }

  /**
   * Показ браузерного уведомления
   */
  showBrowserNotification(title, message, type = 'info') {
    if ('Notification' in window && Notification.permission === 'granted') {
      const icon = this.getNotificationIcon(type)
      
      const notification = new Notification(title, {
        body: message,
        icon: icon,
        badge: icon,
        tag: 'hydraulic-system',
        renotify: true
      })
      
      // Автозакрытие через 5 секунд
      setTimeout(() => notification.close(), 5000)
      
      // Фокус на окне при клике
      notification.onclick = () => {
        window.focus()
        notification.close()
      }
    }
  }

  /**
   * Получение иконки для уведомления
   */
  getNotificationIcon(type) {
    const icons = {
      'info': '/icons/info.png',
      'warning': '/icons/warning.png',
      'error': '/icons/error.png',
      'success': '/icons/success.png'
    }
    return icons[type] || icons.info
  }

  /**
   * Запрос разрешения на уведомления
   */
  async requestNotificationPermission() {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission()
      console.log('🔔 Разрешение на уведомления:', permission)
      return permission === 'granted'
    }
    return false
  }

  /**
   * Heartbeat для поддержания соединения
   */
  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected) {
        this.send({ type: 'ping' })
      }
    }, this.heartbeatTimeout)
  }

  /**
   * Остановка heartbeat
   */
  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }

  /**
   * Обработка переподключения
   */
  handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Превышено максимальное количество попыток переподключения')
      this.emit('reconnect_failed', { attempts: this.reconnectAttempts })
      return
    }
    
    this.reconnectAttempts++
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1) // Экспоненциальный backoff
    
    console.log(`🔄 Попытка переподключения ${this.reconnectAttempts}/${this.maxReconnectAttempts} через ${delay}ms`)
    
    setTimeout(() => {
      this.connect()
    }, delay)
  }

  /**
   * Добавление слушателя событий
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event).push(callback)
  }

  /**
   * Удаление слушателя событий
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event)
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
    }
  }

  /**
   * Генерация события
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error('❌ Ошибка в обработчике события:', error)
        }
      })
    }
  }

  /**
   * Подписка на обновления конкретной системы
   */
  subscribeToSystem(systemId) {
    this.send({
      type: 'subscribe_system',
      system_id: systemId
    })
  }

  /**
   * Отписка от обновлений системы
   */
  unsubscribeFromSystem(systemId) {
    this.send({
      type: 'unsubscribe_system', 
      system_id: systemId
    })
  }

  /**
   * Запрос текущего статуса системы
   */
  requestSystemStatus(systemId) {
    this.send({
      type: 'get_system_status',
      system_id: systemId
    })
  }

  /**
   * Отправка команды системе
   */
  sendSystemCommand(systemId, command, params = {}) {
    this.send({
      type: 'system_command',
      system_id: systemId,
      command: command,
      params: params
    })
  }

  /**
   * Закрытие соединения
   */
  disconnect() {
    console.log('🔌 Закрытие WebSocket соединения')
    this.shouldReconnect = false
    this.stopHeartbeat()
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
    }
    
    // Очистка слушателей
    this.listeners.clear()
    this.messageQueue = []
  }

  /**
   * Получение статуса соединения
   */
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      readyState: this.ws ? this.ws.readyState : WebSocket.CLOSED,
      reconnectAttempts: this.reconnectAttempts
    }
  }

  /**
   * Проверка поддержки WebSocket
   */
  static isSupported() {
    return 'WebSocket' in window
  }
}

// Создание глобального экземпляра
const websocketService = new WebSocketService()

export default websocketService

// Дополнительные утилиты для работы с WebSocket
export class WebSocketHook {
  constructor(wsService) {
    this.wsService = wsService
    this.subscriptions = new Set()
  }

  /**
   * Подписка на события с автоматической очисткой
   */
  useWebSocket(event, callback) {
    this.wsService.on(event, callback)
    this.subscriptions.add({ event, callback })
    
    // Возврат функции отписки
    return () => {
      this.wsService.off(event, callback)
      this.subscriptions.delete({ event, callback })
    }
  }

  /**
   * Очистка всех подписок
   */
  cleanup() {
    this.subscriptions.forEach(({ event, callback }) => {
      this.wsService.off(event, callback)
    })
    this.subscriptions.clear()
  }
}

// Vue композабл для WebSocket
export function useWebSocket() {
  const connectionStatus = ref(websocketService.getConnectionStatus())
  const notifications = ref([])
  
  // Подписка на изменения статуса
  const updateStatus = () => {
    connectionStatus.value = websocketService.getConnectionStatus()
  }
  
  websocketService.on('connected', updateStatus)
  websocketService.on('disconnected', updateStatus)
  websocketService.on('error', updateStatus)
  
  // Обработка уведомлений
  websocketService.on('critical_alert', (alert) => {
    notifications.value.unshift({
      id: Date.now(),
      type: 'critical',
      title: 'Критическое предупреждение',
      message: alert.message,
      systemName: alert.systemName,
      timestamp: new Date()
    })
  })
  
  websocketService.on('diagnostic_completed', (result) => {
    notifications.value.unshift({
      id: Date.now(),
      type: 'info',
      title: 'Диагностика завершена',
      message: 'Анализ системы выполнен',
      timestamp: new Date()
    })
  })
  
  const connect = (token) => {
    websocketService.connect(token)
  }
  
  const disconnect = () => {
    websocketService.disconnect()
  }
  
  const subscribeToSystem = (systemId) => {
    websocketService.subscribeToSystem(systemId)
  }
  
  const unsubscribeFromSystem = (systemId) => {
    websocketService.unsubscribeFromSystem(systemId)
  }
  
  const dismissNotification = (notificationId) => {
    const index = notifications.value.findIndex(n => n.id === notificationId)
    if (index > -1) {
      notifications.value.splice(index, 1)
    }
  }
  
  const clearNotifications = () => {
    notifications.value = []
  }
  
  return {
    connectionStatus: readonly(connectionStatus),
    notifications: readonly(notifications),
    connect,
    disconnect,
    subscribeToSystem,
    unsubscribeFromSystem,
    dismissNotification,
    clearNotifications,
    on: websocketService.on.bind(websocketService),
    off: websocketService.off.bind(websocketService),
    emit: websocketService.emit.bind(websocketService)
  }
}