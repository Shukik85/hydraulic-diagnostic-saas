<template>
  <div class="register-page">
    <div class="register-container">
      <div class="register-card">
        <div class="register-header">
          <h1>🚀 Регистрация</h1>
          <p>Создайте аккаунт для доступа к системе диагностики</p>
        </div>

        <form @submit.prevent="handleRegister" class="register-form">
          <div class="form-row">
            <div class="form-group">
              <label for="firstName">Имя*</label>
              <input 
                id="firstName"
                v-model="formData.first_name" 
                type="text" 
                required
                :class="{ error: errors.first_name }"
              >
              <span v-if="errors.first_name" class="error-text">{{ errors.first_name }}</span>
            </div>
            
            <div class="form-group">
              <label for="lastName">Фамилия*</label>
              <input 
                id="lastName"
                v-model="formData.last_name" 
                type="text" 
                required
                :class="{ error: errors.last_name }"
              >
              <span v-if="errors.last_name" class="error-text">{{ errors.last_name }}</span>
            </div>
          </div>

          <div class="form-group">
            <label for="username">Имя пользователя*</label>
            <input 
              id="username"
              v-model="formData.username" 
              type="text" 
              required
              :class="{ error: errors.username }"
            >
            <span v-if="errors.username" class="error-text">{{ errors.username }}</span>
          </div>

          <div class="form-group">
            <label for="email">Email*</label>
            <input 
              id="email"
              v-model="formData.email" 
              type="email" 
              required
              :class="{ error: errors.email }"
            >
            <span v-if="errors.email" class="error-text">{{ errors.email }}</span>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="company">Компания</label>
              <input 
                id="company"
                v-model="formData.company" 
                type="text"
              >
            </div>
            
            <div class="form-group">
              <label for="position">Должность</label>
              <input 
                id="position"
                v-model="formData.position" 
                type="text"
              >
            </div>
          </div>

          <div class="form-group">
            <label for="password">Пароль*</label>
            <input 
              id="password"
              v-model="formData.password" 
              type="password" 
              required
              :class="{ error: errors.password }"
            >
            <span v-if="errors.password" class="error-text">{{ errors.password }}</span>
          </div>

          <div class="form-group">
            <label for="passwordConfirm">Подтверждение пароля*</label>
            <input 
              id="passwordConfirm"
              v-model="formData.password_confirm" 
              type="password" 
              required
              :class="{ error: errors.password_confirm }"
            >
            <span v-if="errors.password_confirm" class="error-text">{{ errors.password_confirm }}</span>
          </div>

          <div class="form-group checkbox-group">
            <label class="checkbox-label">
              <input 
                v-model="formData.accept_terms" 
                type="checkbox" 
                required
              >
              <span class="checkmark"></span>
              Я согласен с <a href="#" @click.prevent>условиями использования</a> и 
              <a href="#" @click.prevent>политикой конфиденциальности</a>
            </label>
          </div>

          <button 
            type="submit" 
            :disabled="isLoading" 
            class="register-btn"
          >
            <span v-if="isLoading">⏳ Регистрация...</span>
            <span v-else>📝 Зарегистрироваться</span>
          </button>

          <div v-if="errorMessage" class="form-error">
            {{ errorMessage }}
          </div>
        </form>

        <div class="register-footer">
          <p>Уже есть аккаунт? <router-link to="/login">Войти</router-link></p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

export default {
  name: 'Register',
  setup() {
    const router = useRouter()
    
    const formData = ref({
      first_name: '',
      last_name: '',
      username: '',
      email: '',
      company: '',
      position: '',
      password: '',
      password_confirm: '',
      accept_terms: false
    })

    const errors = ref({})
    const isLoading = ref(false)
    const errorMessage = ref('')

    const validateForm = () => {
      errors.value = {}
      let isValid = true

      if (!formData.value.first_name.trim()) {
        errors.value.first_name = 'Имя обязательно для заполнения'
        isValid = false
      }

      if (!formData.value.last_name.trim()) {
        errors.value.last_name = 'Фамилия обязательна для заполнения'
        isValid = false
      }

      if (!formData.value.username.trim()) {
        errors.value.username = 'Имя пользователя обязательно для заполнения'
        isValid = false
      }

      if (!formData.value.email.trim()) {
        errors.value.email = 'Email обязателен для заполнения'
        isValid = false
      } else if (!/\S+@\S+\.\S+/.test(formData.value.email)) {
        errors.value.email = 'Введите корректный email'
        isValid = false
      }

      if (!formData.value.password) {
        errors.value.password = 'Пароль обязателен для заполнения'
        isValid = false
      } else if (formData.value.password.length < 8) {
        errors.value.password = 'Пароль должен содержать минимум 8 символов'
        isValid = false
      }

      if (formData.value.password !== formData.value.password_confirm) {
        errors.value.password_confirm = 'Пароли не совпадают'
        isValid = false
      }

      return isValid
    }

    const handleRegister = async () => {
      if (!validateForm()) return

      isLoading.value = true
      errorMessage.value = ''

      try {
        // Имитация регистрации
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        console.log('Регистрация:', formData.value)
        alert('Регистрация прошла успешно!')
        
        // Перенаправление на страницу входа
        router.push('/login')
        
      } catch (error) {
        console.error('Ошибка регистрации:', error)
        errorMessage.value = 'Ошибка при регистрации. Попробуйте снова.'
      } finally {
        isLoading.value = false
      }
    }

    return {
      formData,
      errors,
      isLoading,
      errorMessage,
      handleRegister
    }
  }
}
</script>

<style scoped>
.register-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.register-container {
  width: 100%;
  max-width: 600px;
}

.register-card {
  background: white;
  border-radius: 16px;
  padding: 3rem;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
}

.register-header {
  text-align: center;
  margin-bottom: 2rem;
}

.register-header h1 {
  font-size: 2.5rem;
  color: #2d3748;
  margin-bottom: 0.5rem;
}

.register-header p {
  color: #64748b;
  font-size: 1.125rem;
}

.register-form {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.form-group {
  display: flex;
  flex-direction: column;
}

.form-group label {
  font-weight: 600;
  color: #374151;
  margin-bottom: 0.5rem;
}

.form-group input {
  padding: 0.75rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.2s;
}

.form-group input:focus {
  outline: none;
  border-color: #667eea;
}

.form-group input.error {
  border-color: #ef4444;
}

.error-text {
  color: #ef4444;
  font-size: 0.875rem;
  margin-top: 0.25rem;
}

.checkbox-group {
  margin: 1rem 0;
}

.checkbox-label {
  display: flex;
  align-items: flex-start;
  cursor: pointer;
  font-size: 0.9rem;
  line-height: 1.5;
}

.checkbox-label input[type="checkbox"] {
  margin-right: 0.75rem;
  margin-top: 0.25rem;
  width: 16px;
  height: 16px;
}

.checkbox-label a {
  color: #667eea;
  text-decoration: none;
}

.checkbox-label a:hover {
  text-decoration: underline;
}

.register-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 1rem;
  border-radius: 8px;
  font-size: 1.125rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.register-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

.register-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.form-error {
  background: #fef2f2;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
  border: 1px solid #fecaca;
}

.register-footer {
  text-align: center;
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid #e2e8f0;
}

.register-footer a {
  color: #667eea;
  text-decoration: none;
  font-weight: 600;
}

.register-footer a:hover {
  text-decoration: underline;
}

@media (max-width: 768px) {
  .register-card {
    padding: 2rem;
  }
  
  .form-row {
    grid-template-columns: 1fr;
  }
  
  .register-header h1 {
    font-size: 2rem;
  }
}
</style>
