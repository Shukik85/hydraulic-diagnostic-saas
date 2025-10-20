<template>
  <div class="systems-list">
    <div class="systems-header">
      <h2>Список гидравлических систем</h2>
      <button @click="refreshSystems" :disabled="loading" class="refresh-btn">
        {{ loading ? 'Загрузка...' : 'Обновить' }}
      </button>
    </div>

    <!-- Фильтры -->
    <div class="filters">
      <input
        v-model="filters.search"
        type="text"
        placeholder="Поиск по названию..."
        class="search-input"
      />
      <select v-model="filters.status" class="status-filter">
        <option value="">Все статусы</option>
        <option value="active">Активна</option>
        <option value="warning">Предупреждение</option>
        <option value="error">Ошибка</option>
        <option value="offline">Отключена</option>
      </select>
    </div>

    <!-- Состояния загрузки и ошибок -->
    <div v-if="loading" class="loading">Загрузка систем...</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <!-- Список систем -->
    <div v-else-if="filteredSystems.length" class="systems-grid">
      <div
        v-for="system in filteredSystems"
        :key="system.id"
        class="system-card"
        :class="`status-${system.status}`"
      >
        <div class="system-info">
          <h3>{{ system.name }}</h3>
          <p class="system-location">{{ system.location }}</p>
        </div>
        <div class="system-status">
          <span class="status-badge" :class="`badge-${system.status}`">
            {{ getStatusLabel(system.status) }}
          </span>
          <span class="last-update">{{ formatDate(system.lastUpdate) }}</span>
        </div>
      </div>
    </div>

    <div v-else class="no-results">
      Системы не найдены
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const { getSystems, systems, loading, error } = useSystems()

// Фильтры
const filters = ref({
  search: '',
  status: ''
})

// Отфильтрованные системы
const filteredSystems = computed(() => {
  let result = systems.value || []

  // Фильтр по поиску
  if (filters.value.search) {
    const searchLower = filters.value.search.toLowerCase()
    result = result.filter(system =>
      system.name.toLowerCase().includes(searchLower) ||
      system.location?.toLowerCase().includes(searchLower)
    )
  }

  // Фильтр по статусу
  if (filters.value.status) {
    result = result.filter(system => system.status === filters.value.status)
  }

  return result
})

// Обновление списка
const refreshSystems = async () => {
  await getSystems()
}

// Форматирование даты
const formatDate = (dateString) => {
  if (!dateString) return 'N/A'
  const date = new Date(dateString)
  return date.toLocaleString('ru-RU', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Метка статуса
const getStatusLabel = (status) => {
  const labels = {
    active: 'Активна',
    warning: 'Предупреждение',
    error: 'Ошибка',
    offline: 'Отключена'
  }
  return labels[status] || 'Неизвестно'
}

// Загрузка при монтировании
onMounted(() => {
  getSystems()
})
</script>

<style scoped>
.systems-list {
  margin-top: 2rem;
  padding: 1.5rem;
}

.systems-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.systems-header h2 {
  color: #2c3e50;
  margin: 0;
}

.refresh-btn {
  padding: 0.5rem 1rem;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.refresh-btn:hover:not(:disabled) {
  background-color: #359268;
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.filters {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.search-input,
.status-filter {
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.95rem;
}

.search-input {
  flex: 1;
}

.status-filter {
  min-width: 150px;
}

.loading,
.error,
.no-results {
  padding: 2rem;
  text-align: center;
  border-radius: 8px;
  background-color: #f9f9f9;
}

.error {
  background-color: #fee;
  color: #c33;
}

.systems-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem;
}

.system-card {
  padding: 1.5rem;
  border-radius: 8px;
  border: 2px solid #ddd;
  background-color: white;
  transition: transform 0.2s, box-shadow 0.2s;
}

.system-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.system-card.status-active {
  border-color: #42b983;
}

.system-card.status-warning {
  border-color: #f39c12;
}

.system-card.status-error {
  border-color: #e74c3c;
}

.system-card.status-offline {
  border-color: #95a5a6;
}

.system-info h3 {
  margin: 0 0 0.5rem 0;
  color: #2c3e50;
}

.system-location {
  margin: 0;
  color: #7f8c8d;
  font-size: 0.9rem;
}

.system-status {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #ecf0f1;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
}

.badge-active {
  background-color: #d4edda;
  color: #155724;
}

.badge-warning {
  background-color: #fff3cd;
  color: #856404;
}

.badge-error {
  background-color: #f8d7da;
  color: #721c24;
}

.badge-offline {
  background-color: #e2e3e5;
  color: #383d41;
}

.last-update {
  font-size: 0.8rem;
  color: #95a5a6;
}
</style>
