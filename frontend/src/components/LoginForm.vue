<template>
  <div class="login-container">
    <div class="login-form">
      <div class="form-header">
        <h2>🔧 Вход в систему</h2>
        <p>Диагностика гидравлических систем</p>
      </div>

      <form @submit.prevent="handleLogin" class="form">
        <div class="form-group">
          <label for="email">Email</label>
          <input
            id="email"
            v-model="form.email"
            type="email"
            :class="{ error: errors.email }"
            placeholder="Введите ваш email"
            required
          />
          <span v-if="errors.email" class="error-message">{{ errors.email }}</span>
        </div>

        <div class="form-group">
          <label for="password">Пароль</label>
          <input
            id="password"
            v-model="form.password"
            type="password"
            :class="{ error: errors.password }"
            placeholder="Введите пароль"
            required
          />
          <span v-if="errors.password" class="error-message">{{ errors.password }}</span>
        </div>

        <button type="submit" class="btn btn-primary" :disabled="loading">
          {{ loading ? 'Вход...' : 'Войти' }}
        </button>

        <div v-if="submitError" class="error-banner">
          {{ submitError }}
        </div>
      </form>

      <div class="form-footer">
        <p>Нет аккаунта? <a href="#" @click="$emit('switchToRegister')">Регистрация</a></p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive } from 'vue'
import { authService } from '@/services/authService'

export default {
  name: 'LoginForm',
  emits: ['switchToRegister', 'loginSuccess'],
  setup(props, { emit }) {
    const loading = ref(false)
    const submitError = ref('')

    const form = reactive({
      email: '',
      password: ''
    })

    const errors = reactive({})

    const validateForm = () => {
      Object.keys(errors).forEach(key => delete errors[key])

      if (!form.email) {
        errors.email = 'Email обязателен'
      }

      if (!form.password) {
        errors.password = 'Пароль обязателен'
      }

      return Object.keys(errors).length === 0
    }

    const handleLogin = async () => {
      if (!validateForm()) {
        return
      }

      loading.value = true
      submitError.value = ''

      try {
        const result = await authService.login({
          username: form.email, // Django ожидает username
          password: form.password
        })
        
        emit('loginSuccess', result.user)
      } catch (error) {
        console.error('Ошибка входа:', error)
        if (error.response?.data?.detail) {
          submitError.value = error.response.data.detail
        } else {
          submitError.value = 'Ошибка входа в систему'
        }
      } finally {
        loading.value = false
      }
    }

    return {
      form,
      errors,
      loading,
      submitError,
      handleLogin
    }
  }
}
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 1rem;
}

.login-form {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  width: 100%;
  max-width: 400px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.form-header {
  text-align: center;
  margin-bottom: 2rem;
}

.form-header h2 {
  color: #1f2937;
  margin-bottom: 0.5rem;
}

.form-header p {
  color: #6b7280;
  font-size: 0.875rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #374151;
}

.form-group input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.875rem;
  transition: border-color 0.2s;
}

.form-group input:focus {
  outline: none;
  border-color: #3b82f6;
}

.form-group input.error {
  border-color: #dc2626;
}

.error-message {
  color: #dc2626;
  font-size: 0.75rem;
  margin-top: 0.25rem;
}

.btn-primary {
  width: 100%;
  padding: 0.75rem;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
  margin-top: 1rem;
}

.btn-primary:hover {
  background: #2563eb;
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-banner {
  background: #fef2f2;
  color: #dc2626;
  padding: 0.75rem;
  border-radius: 6px;
  font-size: 0.875rem;
  margin-top: 1rem;
  text-align: center;
}

.form-footer {
  text-align: center;
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.form-footer a {
  color: #3b82f6;
  text-decoration: none;
}

.form-footer a:hover {
  text-decoration: underline;
}
</style>
