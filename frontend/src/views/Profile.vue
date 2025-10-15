<template>
  <div class="profile-page">
    <div class="page-header">
      <h1>👤 Профиль пользователя</h1>
      <p>Управление личными данными и настройками</p>
    </div>

    <div class="profile-content">
      <div class="profile-sections">
        <div class="profile-info">
          <h2>Основная информация</h2>
          <form @submit.prevent="updateProfile">
            <div class="form-row">
              <div class="form-group">
                <label for="firstName">Имя:</label>
                <input 
                  id="firstName"
                  v-model="profile.first_name" 
                  type="text" 
                  required
                >
              </div>
              
              <div class="form-group">
                <label for="lastName">Фамилия:</label>
                <input 
                  id="lastName"
                  v-model="profile.last_name" 
                  type="text" 
                  required
                >
              </div>
            </div>

            <div class="form-group">
              <label for="email">Email:</label>
              <input 
                id="email"
                v-model="profile.email" 
                type="email" 
                required
              >
            </div>

            <div class="form-row">
              <div class="form-group">
                <label for="company">Компания:</label>
                <input 
                  id="company"
                  v-model="profile.company" 
                  type="text"
                >
              </div>
              
              <div class="form-group">
                <label for="position">Должность:</label>
                <input 
                  id="position"
                  v-model="profile.position" 
                  type="text"
                >
              </div>
            </div>

            <div class="form-group">
              <label for="phone">Телефон:</label>
              <input 
                id="phone"
                v-model="profile.phone" 
                type="tel"
              >
            </div>

            <button type="submit" class="btn btn-primary">
              Сохранить изменения
            </button>
          </form>
        </div>

        <div class="profile-settings">
          <h2>Настройки уведомлений</h2>
          <div class="settings-list">
            <div class="setting-item">
              <label class="checkbox-label">
                <input 
                  v-model="profile.email_notifications" 
                  type="checkbox"
                >
                <span class="checkmark"></span>
                Email уведомления
              </label>
            </div>
            
            <div class="setting-item">
              <label class="checkbox-label">
                <input 
                  v-model="profile.push_notifications" 
                  type="checkbox"
                >
                <span class="checkmark"></span>
                Push уведомления
              </label>
            </div>
            
            <div class="setting-item">
              <label class="checkbox-label">
                <input 
                  v-model="profile.critical_alerts_only" 
                  type="checkbox"
                >
                <span class="checkmark"></span>
                Только критичные уведомления
              </label>
            </div>
          </div>
        </div>

        <div class="profile-stats">
          <h2>Статистика</h2>
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-value">{{ profile.systems_count || 0 }}</div>
              <div class="stat-label">Систем</div>
            </div>
            
            <div class="stat-card">
              <div class="stat-value">{{ profile.reports_generated || 0 }}</div>
              <div class="stat-label">Отчетов создано</div>
            </div>
            
            <div class="stat-card">
              <div class="stat-value">{{ formatDate(profile.created_at) }}</div>
              <div class="stat-label">Дата регистрации</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'

export default {
  name: 'Profile',
  setup() {
    const profile = ref({
      first_name: 'Иван',
      last_name: 'Петров',
      email: 'ivan.petrov@example.com',
      company: 'ООО "Гидротех"',
      position: 'Инженер-диагност',
      phone: '+7 (999) 123-45-67',
      email_notifications: true,
      push_notifications: true,
      critical_alerts_only: false,
      systems_count: 5,
      reports_generated: 23,
      created_at: '2023-01-15T10:30:00Z'
    })

    const loadProfile = async () => {
      // Здесь будет загрузка профиля из API
      console.log('Загрузка профиля пользователя')
    }

    const updateProfile = async () => {
      try {
        // Здесь будет отправка обновленного профиля на сервер
        console.log('Обновление профиля:', profile.value)
        alert('Профиль успешно обновлен!')
      } catch (error) {
        console.error('Ошибка обновления профиля:', error)
        alert('Ошибка при сохранении изменений')
      }
    }

    const formatDate = (dateString) => {
      if (!dateString) return 'Не указано'
      return new Date(dateString).toLocaleDateString('ru-RU')
    }

    onMounted(() => {
      loadProfile()
    })

    return {
      profile,
      updateProfile,
      formatDate
    }
  }
}
</script>

<style scoped>
.profile-page {
  padding: 2rem;
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  margin-bottom: 3rem;
}

.page-header h1 {
  font-size: 2.5rem;
  color: #2d3748;
  margin-bottom: 0.5rem;
}

.profile-sections {
  display: grid;
  gap: 2rem;
}

.profile-info,
.profile-settings,
.profile-stats {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  font-weight: 600;
  color: #374151;
  margin-bottom: 0.5rem;
}

.form-group input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.2s;
}

.form-group input:focus {
  outline: none;
  border-color: #667eea;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.settings-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.setting-item {
  display: flex;
  align-items: center;
}

.checkbox-label {
  display: flex;
  align-items: center;
  cursor: pointer;
  font-weight: 500;
}

.checkbox-label input[type="checkbox"] {
  margin-right: 0.75rem;
  width: 18px;
  height: 18px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1.5rem;
}

.stat-card {
  text-align: center;
  padding: 1.5rem;
  background: #f8fafc;
  border-radius: 12px;
}

.stat-value {
  font-size: 2rem;
  font-weight: bold;
  color: #667eea;
  margin-bottom: 0.5rem;
}

.stat-label {
  color: #64748b;
  font-weight: 500;
}

@media (max-width: 768px) {
  .form-row {
    grid-template-columns: 1fr;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>
