<template>
  <div class="login-container">
    <div class="login-card">
      <h1>Вход в систему</h1>
      
      <form @submit.prevent="handleLogin">
        <div class="form-group">
          <label for="email">Email:</label>
          <input 
            id="email"
            v-model="credentials.email" 
            type="email" 
            required
            :disabled="loading"
          />
        </div>
        
        <div class="form-group">
          <label for="password">Пароль:</label>
          <input 
            id="password"
            v-model="credentials.password" 
            type="password" 
            required
            :disabled="loading"
          />
        </div>
        
        <div v-if="error" class="error-message">
          {{ error }}
        </div>
        
        <button type="submit" :disabled="loading">
          {{ loading ? 'Вход...' : 'Войти' }}
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
const { login, isAuthenticated } = useAuth()
const router = useRouter()

const credentials = ref({
  email: '',
  password: ''
})

const loading = ref(false)
const error = ref('')

// Redirect if already authenticated
onMounted(() => {
  if (isAuthenticated.value) {
    router.push('/dashboard')
  }
})

const handleLogin = async () => {
  error.value = ''
  loading.value = true
  
  try {
    await login(credentials.value)
    router.push('/dashboard')
  } catch (err) {
    error.value = err.message || 'Ошибка входа. Проверьте данные.'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: #f5f5f5;
}

.login-card {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

h1 {
  margin-bottom: 1.5rem;
  text-align: center;
  color: #333;
}

.form-group {
  margin-bottom: 1rem;
}

label {
  display: block;
  margin-bottom: 0.5rem;
  color: #555;
  font-weight: 500;
}

input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  transition: border-color 0.3s;
}

input:focus {
  outline: none;
  border-color: #4CAF50;
}

input:disabled {
  background: #f9f9f9;
  cursor: not-allowed;
}

button {
  width: 100%;
  padding: 0.75rem;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.3s;
}

button:hover:not(:disabled) {
  background: #45a049;
}

button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.error-message {
  padding: 0.75rem;
  background: #ffebee;
  color: #c62828;
  border-radius: 4px;
  margin-bottom: 1rem;
  font-size: 0.9rem;
}
</style>
