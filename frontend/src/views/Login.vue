<template>
  <div class="login-page">
    <div class="login-form">
      <h2>🔧 Вход в систему</h2>
      
      <form @submit.prevent="handleLogin">
        <div class="form-group">
          <input v-model="form.username" type="email" placeholder="Email" required />
        </div>
        
        <div class="form-group">
          <input v-model="form.password" type="password" placeholder="Пароль" required />
        </div>
        
        <button type="submit" :disabled="loading">
          {{ loading ? 'Вход...' : 'Войти' }}
        </button>
        
        <p v-if="error" class="error">{{ error }}</p>
      </form>
      
      <div class="register-link">
        <p>
          Нет аккаунта? 
          <button @click="showRegister = !showRegister" class="link-btn">
            Регистрация
          </button>
        </p>
      </div>
      
      <div v-if="showRegister" class="register-form">
        <h3>Регистрация</h3>
        <form @submit.prevent="handleRegister">
          <input v-model="regForm.email" type="email" placeholder="Email" required />
          <input v-model="regForm.username" type="text" placeholder="Имя пользователя" required />
          <input v-model="regForm.password" type="password" placeholder="Пароль" required />
          <input v-model="regForm.password_confirm" type="password" placeholder="Подтвердите пароль" required />
          <button type="submit">Зарегистрироваться</button>
        </form>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'  // Добавь этот импорт
import { authService } from '@/services/authService'

export default {
  name: 'Login',
  setup() {
    const router = useRouter()  // Добавь эту строку
    const loading = ref(false)
    const error = ref('')
    const showRegister = ref(false)
    
    const form = reactive({
      username: '',
      password: ''
    })
    
    const regForm = reactive({
      email: '',
      username: '',
      password: '',
      password_confirm: ''
    })

    const handleLogin = async () => {
      loading.value = true
      error.value = ''
      
      try {
        await authService.login(form)
        router.push('/')
      } catch (err) {
        error.value = err.response?.data?.detail || 'Ошибка входа'
        console.error('Login error:', err)
      } finally {
        loading.value = false
      }
    }

    const handleRegister = async () => {
      if (regForm.password !== regForm.password_confirm) {
        error.value = 'Пароли не совпадают'
        return
      }
      
      try {
        await authService.register(regForm)
        alert('Регистрация успешна! Теперь войдите.')
        showRegister.value = false
        
        // Очистить форму регистрации
        Object.keys(regForm).forEach(key => regForm[key] = '')
      } catch (err) {
        error.value = err.response?.data?.detail || 'Ошибка регистрации'
        console.error('Registration error:', err)
      }
    }

    return {
      form,
      regForm,
      loading,
      error,
      showRegister,
      handleLogin,
      handleRegister
    }
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 1rem;
}

.login-form {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  width: 100%;
  max-width: 400px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.login-form h2 {
  text-align: center;
  margin-bottom: 1.5rem;
  color: #1f2937;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 1rem;
}

.form-group input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

button[type="submit"] {
  width: 100%;
  padding: 0.75rem;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.2s;
}

button[type="submit"]:hover {
  background: #2563eb;
}

button[type="submit"]:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error {
  color: #dc2626;
  text-align: center;
  margin-top: 1rem;
  font-size: 0.9rem;
}

.register-link {
  text-align: center;
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.link-btn {
  background: none;
  border: none;
  color: #3b82f6;
  cursor: pointer;
  text-decoration: underline;
  font-size: inherit;
}

.link-btn:hover {
  color: #2563eb;
}

.register-form {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.register-form h3 {
  margin-bottom: 1rem;
  color: #1f2937;
}

.register-form input {
  width: 100%;
  padding: 0.5rem;
  margin-bottom: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 4px;
}

.register-form button {
  width: 100%;
  padding: 0.75rem;
  background: #10b981;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

.register-form button:hover {
  background: #059669;
}
</style>
