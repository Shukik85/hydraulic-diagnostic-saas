<template>
  <div>
    <Login v-if="!isAuthenticated" />
    <Dashboard v-else />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import Login from './Login.vue'
import Dashboard from './dashboard.vue'
import { useAuth } from '~/composables/useAuth'

const { getToken } = useAuth()
const isAuthenticated = ref(false)

onMounted(() => {
  const token = getToken()
  isAuthenticated.value = !!token
})
</script>
