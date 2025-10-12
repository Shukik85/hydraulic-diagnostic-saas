<template>
  <div class="dashboard">
    <header class="header">
      <div class="container">
        <h1>🔧 Диагностика гидравлических систем</h1>
        <nav>
          <button class="btn">Системы</button>
          <button class="btn btn-secondary">Отчеты</button>
        </nav>
      </div>
    </header>

    <main class="main">
      <div class="container">
        <div class="welcome-card card">
          <h2>Добро пожаловать в систему диагностики!</h2>
          <p>Система успешно запущена и готова к работе.</p>
          
          <div class="status-grid">
            <div class="status-item">
              <div class="status-icon">🚀</div>
              <div>
                <h3>Django Backend</h3>
                <p class="status-ok">Работает</p>
              </div>
            </div>
            
            <div class="status-item">
              <div class="status-icon">🗄️</div>
              <div>
                <h3>PostgreSQL</h3>
                <p class="status-ok">Подключено</p>
              </div>
            </div>
            
            <div class="status-item">
              <div class="status-icon">⚡</div>
              <div>
                <h3>Vue.js Frontend</h3>
                <p class="status-ok">Активен</p>
              </div>
            </div>
          </div>

          <div class="actions">
            <button class="btn" @click="testAPI">Тест API</button>
            <button class="btn btn-secondary" @click="showInfo">Информация</button>
          </div>

          <div v-if="apiResponse" class="api-response">
            <h4>Ответ API:</h4>
            <pre>{{ apiResponse }}</pre>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'

export default {
  name: 'Dashboard',
  setup() {
    const apiResponse = ref(null)

    const testAPI = async () => {
      try {
        // Пока просто показываем тестовый ответ
        apiResponse.value = {
          status: 'success',
          message: 'API готово к работе',
          timestamp: new Date().toLocaleString()
        }
      } catch (error) {
        apiResponse.value = {
          status: 'error',
          message: error.message
        }
      }
    }

    const showInfo = () => {
      alert('MVP диагностической системы v1.0\nРазработка: 12 октября 2025')
    }

    return {
      apiResponse,
      testAPI,
      showInfo
    }
  }
}
</script>

<style scoped>
.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 0;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header .container {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header h1 {
  font-size: 1.5rem;
  margin: 0;
}

.header nav {
  display: flex;
  gap: 10px;
}

.main {
  padding: 2rem 0;
}

.welcome-card {
  text-align: center;
  max-width: 800px;
  margin: 0 auto;
}

.welcome-card h2 {
  color: #2563eb;
  margin-bottom: 1rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
}

.status-icon {
  font-size: 2rem;
}

.status-ok {
  color: #10b981;
  font-weight: bold;
}

.actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin: 2rem 0;
}

.api-response {
  background: #f1f5f9;
  padding: 1rem;
  border-radius: 6px;
  margin-top: 1rem;
  text-align: left;
}

.api-response pre {
  background: #1f2937;
  color: #10b981;
  padding: 1rem;
  border-radius: 4px;
  overflow-x: auto;
}
</style>
