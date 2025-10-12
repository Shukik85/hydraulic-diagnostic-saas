<template>
  <div class="dashboard">
    <header class="header">
      <div class="container">
        <h1>🔧 Диагностика гидравлических систем</h1>
        <div class="user-info">
          Привет, {{ user?.username }}!
          <button @click="handleLogout" class="btn">Выход</button>
        </div>
      </div>
    </header>

    <main class="main">
      <div class="container">
        <SystemsList />
      </div>
    </main>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'  // Добавь этот импорт
import { authService } from '@/services/authService'
import SystemsList from '@/components/SystemsList.vue'

export default {
  name: 'Dashboard',
  components: {
    SystemsList
  },
  setup() {
    const router = useRouter()  // Добавь эту строку
    const user = ref(null)

    const checkAuth = () => {
      if (!authService.isAuthenticated()) {
        router.push('/login')  // Замени window.location.href на это
        return
      }
      user.value = authService.getCurrentUser()
    }

    const handleLogout = () => {
      authService.logout()
      router.push('/login')  // Замени window.location.href на это
    }

    onMounted(() => {
      checkAuth()
    })

    return {
      user,
      handleLogout
    }
  }
}
</script>

<style scoped>
.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 0;
}

.header .container {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.main {
  padding: 2rem 0;
}
</style>
