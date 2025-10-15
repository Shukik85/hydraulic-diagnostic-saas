<template>
  <div id="app" class="app-container">
    <!-- Навигационная панель -->
    <nav class="navbar">
      <div class="nav-container">
        <!-- Логотип и название -->
        <div class="nav-brand">
          <router-link to="/" class="brand-link">
            <div class="brand-icon">🔧</div>
            <div class="brand-text">
              <div class="brand-title">HydroSys</div>
              <div class="brand-subtitle">Диагностический комплекс</div>
            </div>
          </router-link>
        </div>

        <!-- Основные навигационные ссылки -->
        <div class="nav-menu" v-if="isAuthenticated">
          <router-link to="/" class="nav-item" active-class="nav-item-active">
            <i class="nav-icon">🏠</i>
            <span>Дашборд</span>
          </router-link>
          
          <router-link to="/systems" class="nav-item" active-class="nav-item-active">
            <i class="nav-icon">⚙️</i>
            <span>Системы</span>
          </router-link>
          
          <router-link to="/analytics" class="nav-item" active-class="nav-item-active">
            <i class="nav-icon">📊</i>
            <span>Аналитика</span>
          </router-link>
          
          <router-link to="/reports" class="nav-item" active-class="nav-item-active">
            <i class="nav-icon">📋</i>
            <span>Отчеты</span>
          </router-link>
          
          <div class="nav-item nav-dropdown" @click="toggleAIMenu">
            <i class="nav-icon">🤖</i>
            <span>AI Помощник</span>
            <i class="dropdown-arrow">▼</i>
            
            <div class="dropdown-menu" v-show="showAIMenu">
              <router-link to="/ai-chat" class="dropdown-item">
                💬 Чат с AI
              </router-link>
              <router-link to="/knowledge-base" class="dropdown-item">
                📚 База знаний
              </router-link>
              <router-link to="/diagnostics" class="dropdown-item">
                🔍 AI Диагностика
              </router-link>
            </div>
          </div>
        </div>

        <!-- Пользовательское меню -->
        <div class="nav-user" v-if="isAuthenticated">
          <!-- Уведомления -->
          <div class="notification-bell" @click="toggleNotifications">
            <i class="bell-icon">🔔</i>
            <span class="notification-count" v-if="notificationCount > 0">
              {{ notificationCount }}
            </span>
          </div>
          
          <!-- Пользовательское меню -->
          <div class="user-menu" @click="toggleUserMenu">
            <div class="user-avatar">
              <i class="avatar-icon">👤</i>
            </div>
            <div class="user-info">
              <div class="user-name">{{ user?.username }}</div>
              <div class="user-email">{{ user?.email }}</div>
            </div>
            <i class="dropdown-arrow">▼</i>
            
            <div class="dropdown-menu user-dropdown" v-show="showUserMenu">
              <router-link to="/profile" class="dropdown-item">
                ⚙️ Настройки профиля
              </router-link>
              <div class="dropdown-divider"></div>
              <a href="#" @click.prevent="handleLogout" class="dropdown-item">
                🚪 Выход
              </a>
            </div>
          </div>
        </div>

        <!-- Кнопка входа для неавторизованных -->
        <div class="nav-auth" v-else>
          <router-link to="/login" class="btn btn-primary">Вход</router-link>
        </div>
      </div>
    </nav>

    <!-- Боковая панель быстрого доступа (опционально) -->
    <div class="sidebar" v-if="showSidebar && isAuthenticated">
      <div class="sidebar-content">
        <div class="sidebar-section">
          <h3>Быстрый доступ</h3>
          <div class="quick-actions">
            <button class="quick-action-btn" @click="$emit('create-system')">
              ➕ Создать систему
            </button>
            <button class="quick-action-btn" @click="$emit('run-diagnostics')">
              🔍 Диагностика
            </button>
            <button class="quick-action-btn" @click="$emit('view-alerts')">
              ⚠️ Предупреждения
            </button>
          </div>
        </div>
        
        <div class="sidebar-section" v-if="recentSystems.length > 0">
          <h3>Недавние системы</h3>
          <div class="recent-systems">
            <router-link 
              v-for="system in recentSystems" 
              :key="system.id"
              :to="`/systems/${system.id}`"
              class="recent-system-item"
            >
              <div class="system-status" :class="`status-${system.status}`"></div>
              <span>{{ system.name }}</span>
            </router-link>
          </div>
        </div>
      </div>
    </div>

    <!-- Основной контент -->
    <main class="main-content" :class="{ 'with-sidebar': showSidebar && isAuthenticated }">
      <router-view />
    </main>

    <!-- Уведомления -->
    <div class="notifications-panel" v-show="showNotifications">
      <div class="notifications-header">
        <h3>Уведомления</h3>
        <button @click="toggleNotifications" class="close-btn">✕</button>
      </div>
      
      <div class="notifications-content">
        <div v-if="notifications.length === 0" class="no-notifications">
          Нет новых уведомлений
        </div>
        
        <div 
          v-for="notification in notifications" 
          :key="notification.id"
          class="notification-item"
          :class="`notification-${notification.type}`"
        >
          <div class="notification-icon">
            {{ getNotificationIcon(notification.type) }}
          </div>
          <div class="notification-content">
            <div class="notification-title">{{ notification.title }}</div>
            <div class="notification-message">{{ notification.message }}</div>
            <div class="notification-time">{{ formatTime(notification.timestamp) }}</div>
          </div>
          <button @click="dismissNotification(notification.id)" class="dismiss-btn">
            ✕
          </button>
        </div>
      </div>

    <!-- Загрузчик -->
    <div class="app-loader" v-if="isLoading">
      <div class="loader-spinner"></div>
      <div class="loader-text">Загрузка...</div>
    </div>
  </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { authService } from '@/services/authService'
import { hydraulicSystemService } from '@/services/hydraulicSystemService'

export default {
  name: 'MainLayout',
  emits: ['create-system', 'run-diagnostics', 'view-alerts'],
  setup(props, { emit }) {
    const router = useRouter()
    const route = useRoute()
    
    // Реактивные данные
    const isLoading = ref(false)
    const showSidebar = ref(true)
    const showUserMenu = ref(false)
    const showAIMenu = ref(false)
    const showNotifications = ref(false)
    const user = ref(null)
    const recentSystems = ref([])
    const notifications = ref([])
    
    // Вычисляемые свойства
    const isAuthenticated = computed(() => authService.isAuthenticated())
    const notificationCount = computed(() => 
      notifications.value.filter(n => !n.dismissed).length
    )
    
    // Методы
    const toggleUserMenu = () => {
      showUserMenu.value = !showUserMenu.value
      if (showUserMenu.value) {
        showAIMenu.value = false
        showNotifications.value = false
      }
    }
    
    const toggleAIMenu = () => {
      showAIMenu.value = !showAIMenu.value
      if (showAIMenu.value) {
        showUserMenu.value = false
        showNotifications.value = false
      }
    }
    
    const toggleNotifications = () => {
      showNotifications.value = !showNotifications.value
      if (showNotifications.value) {
        showUserMenu.value = false
        showAIMenu.value = false
      }
    }
    
    const handleLogout = () => {
      authService.logout()
      router.push('/login')
    }
    
    const loadUserData = () => {
      user.value = authService.getCurrentUser()
    }
    
    const loadRecentSystems = async () => {
      try {
        const response = await hydraulicSystemService.getSystems()
        const systems = Array.isArray(response) ? response : response.results || []
        recentSystems.value = systems.slice(0, 5) // Последние 5 систем
      } catch (error) {
        console.error('Ошибка загрузки систем:', error)
      }
    }
    
    const getNotificationIcon = (type) => {
      const icons = {
        'critical': '🚨',
        'warning': '⚠️',
        'info': 'ℹ️',
        'success': '✅'
      }
      return icons[type] || 'ℹ️'
    }
    
    const formatTime = (timestamp) => {
      return new Date(timestamp).toLocaleString('ru-RU', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
    
    const dismissNotification = (id) => {
      const index = notifications.value.findIndex(n => n.id === id)
      if (index !== -1) {
        notifications.value[index].dismissed = true
      }
    }
    
    const loadNotifications = () => {
      // Симуляция загрузки уведомлений
      // В реальном приложении это будет API вызов
      notifications.value = [
        {
          id: 1,
          type: 'critical',
          title: 'Критическое давление',
          message: 'Система "Пресс №1" превысила максимальное давление',
          timestamp: new Date().toISOString(),
          dismissed: false
        },
        {
          id: 2,
          type: 'warning',
          title: 'Высокая температура',
          message: 'Температура масла в системе "Кран №2" выше нормы',
          timestamp: new Date(Date.now() - 30*60*1000).toISOString(),
          dismissed: false
        }
      ]
    }
    
    // Закрытие меню при клике вне их
    const handleOutsideClick = (event) => {
      if (!event.target.closest('.nav-dropdown') && 
          !event.target.closest('.user-menu') && 
          !event.target.closest('.notifications-panel')) {
        showUserMenu.value = false
        showAIMenu.value = false
        showNotifications.value = false
      }
    }
    
    // Жизненный цикл
    onMounted(() => {
      if (isAuthenticated.value) {
        loadUserData()
        loadRecentSystems()
        loadNotifications()
      }
      
      document.addEventListener('click', handleOutsideClick)
    })
    
    // Отслеживание изменений аутентификации
    watch(() => route.path, () => {
      if (isAuthenticated.value && !user.value) {
        loadUserData()
        loadRecentSystems()
        loadNotifications()
      }
    })
    
    return {
      isLoading,
      showSidebar,
      showUserMenu,
      showAIMenu,
      showNotifications,
      user,
      recentSystems,
      notifications,
      isAuthenticated,
      notificationCount,
      toggleUserMenu,
      toggleAIMenu,
      toggleNotifications,
      handleLogout,
      getNotificationIcon,
      formatTime,
      dismissNotification
    }
  }
}
</script>

<style scoped>
/* Основные стили приложения */
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  position: relative;
}

/* Навигационная панель */
.navbar {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
}

/* Брендинг */
.nav-brand {
  display: flex;
  align-items: center;
}

.brand-link {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: inherit;
  gap: 0.75rem;
}

.brand-icon {
  font-size: 2rem;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.brand-title {
  font-size: 1.5rem;
  font-weight: bold;
  color: #2d3748;
}

.brand-subtitle {
  font-size: 0.75rem;
  color: #718096;
  margin-top: -2px;
}

/* Навигационное меню */
.nav-menu {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  text-decoration: none;
  color: #4a5568;
  border-radius: 8px;
  transition: all 0.2s;
  position: relative;
  cursor: pointer;
}

.nav-item:hover {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
  transform: translateY(-1px);
}

.nav-item-active {
  background: rgba(102, 126, 234, 0.15);
  color: #667eea;
  font-weight: 600;
}

.nav-icon {
  font-size: 1.1rem;
}

/* Dropdown меню */
.nav-dropdown {
  position: relative;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  background: white;
  border-radius: 8px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  min-width: 200px;
  z-index: 1000;
  overflow: hidden;
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.dropdown-item {
  display: block;
  padding: 0.75rem 1rem;
  text-decoration: none;
  color: #4a5568;
  transition: background 0.2s;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.dropdown-item:hover {
  background: #f7fafc;
  color: #667eea;
}

.dropdown-item:last-child {
  border-bottom: none;
}

.dropdown-divider {
  height: 1px;
  background: rgba(0, 0, 0, 0.1);
  margin: 0.5rem 0;
}

/* Пользовательская область */
.nav-user {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.notification-bell {
  position: relative;
  padding: 0.5rem;
  cursor: pointer;
  border-radius: 50%;
  transition: background 0.2s;
}

.notification-bell:hover {
  background: rgba(102, 126, 234, 0.1);
}

.bell-icon {
  font-size: 1.25rem;
  color: #4a5568;
}

.notification-count {
  position: absolute;
  top: 0;
  right: 0;
  background: #e53e3e;
  color: white;
  border-radius: 50%;
  width: 18px;
  height: 18px;
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
}

.user-menu {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.user-menu:hover {
  background: rgba(102, 126, 234, 0.1);
  transform: translateY(-1px);
}

.user-avatar {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 1.1rem;
}

.user-info {
  display: flex;
  flex-direction: column;
}

.user-name {
  font-weight: 600;
  color: #2d3748;
  font-size: 0.9rem;
}

.user-email {
  color: #718096;
  font-size: 0.8rem;
}

.user-dropdown {
  right: 0;
  left: auto;
}

/* Кнопки */
.btn {
  display: inline-block;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  text-decoration: none;
  font-weight: 500;
  transition: all 0.2s;
  border: none;
  cursor: pointer;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

/* Боковая панель */
.sidebar {
  position: fixed;
  left: 0;
  top: 80px;
  width: 280px;
  height: calc(100vh - 80px);
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(0, 0, 0, 0.1);
  z-index: 50;
  overflow-y: auto;
}

.sidebar-content {
  padding: 1.5rem;
}

.sidebar-section {
  margin-bottom: 2rem;
}

.sidebar-section h3 {
  color: #2d3748;
  margin-bottom: 1rem;
  font-size: 1rem;
  font-weight: 600;
}

.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.quick-action-btn {
  padding: 0.75rem;
  background: rgba(102, 126, 234, 0.1);
  border: none;
  border-radius: 8px;
  color: #667eea;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  font-weight: 500;
}

.quick-action-btn:hover {
  background: rgba(102, 126, 234, 0.2);
  transform: translateX(5px);
}

.recent-systems {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.recent-system-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  text-decoration: none;
  color: #4a5568;
  border-radius: 8px;
  transition: all 0.2s;
}

.recent-system-item:hover {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
}

.system-status {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-active { background: #48bb78; }
.status-maintenance { background: #ed8936; }
.status-inactive { background: #a0aec0; }
.status-faulty { background: #f56565; }

/* Основной контент */
.main-content {
  flex: 1;
  padding: 2rem;
  transition: margin-left 0.3s;
}

.main-content.with-sidebar {
  margin-left: 280px;
}

/* Панель уведомлений */
.notifications-panel {
  position: fixed;
  top: 80px;
  right: 20px;
  width: 350px;
  max-height: 60vh;
  background: white;
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  z-index: 1000;
  overflow: hidden;
}

.notifications-header {
  padding: 1rem 1.5rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  background: #f7fafc;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.notifications-header h3 {
  margin: 0;
  color: #2d3748;
  font-size: 1rem;
}

.notifications-content {
  max-height: 400px;
  overflow-y: auto;
}

.no-notifications {
  padding: 2rem;
  text-align: center;
  color: #718096;
}

.notification-item {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  transition: background 0.2s;
}

.notification-item:hover {
  background: #f7fafc;
}

.notification-critical {
  border-left: 4px solid #f56565;
}

.notification-warning {
  border-left: 4px solid #ed8936;
}

.notification-info {
  border-left: 4px solid #4299e1;
}

.notification-success {
  border-left: 4px solid #48bb78;
}

.notification-icon {
  font-size: 1.25rem;
  margin-top: 0.25rem;
}

.notification-content {
  flex: 1;
}

.notification-title {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.notification-message {
  color: #4a5568;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.notification-time {
  color: #718096;
  font-size: 0.8rem;
}

.dismiss-btn, .close-btn {
  background: none;
  border: none;
  color: #a0aec0;
  cursor: pointer;
  font-size: 1rem;
  padding: 0.25rem;
  border-radius: 4px;
  transition: color 0.2s;
}

.dismiss-btn:hover, .close-btn:hover {
  color: #4a5568;
  background: rgba(0, 0, 0, 0.05);
}

/* Загрузчик */
.app-loader {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 2000;
}

.loader-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #e2e8f0;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loader-text {
  margin-top: 1rem;
  color: #4a5568;
  font-weight: 500;
}

/* Адаптивность */
@media (max-width: 768px) {
  .nav-container {
    padding: 0.5rem;
  }
  
  .nav-menu {
    display: none; /* Скрыть на мобильных, добавить гамбургер меню */
  }
  
  .sidebar {
    transform: translateX(-100%);
    transition: transform 0.3s;
  }
  
  .sidebar.open {
    transform: translateX(0);
  }
  
  .main-content {
    margin-left: 0;
    padding: 1rem;
  }
  
  .notifications-panel {
    width: calc(100vw - 40px);
    right: 20px;
    left: 20px;
  }
  
  .brand-subtitle {
    display: none;
  }
  
  .user-info {
    display: none;
  }
}

@media (max-width: 480px) {
  .nav-container {
    flex-wrap: wrap;
  }
  
  .nav-user {
    gap: 0.5rem;
  }
  
  .user-avatar {
    width: 32px;
    height: 32px;
    font-size: 0.9rem;
  }
}
</style>
