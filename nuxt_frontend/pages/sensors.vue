<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div>
        <h1 class="u-h2">Датчики</h1>
        <p class="u-body text-gray-600 mt-1">
          Мониторинг и управление датчиками гидравлических систем
        </p>
      </div>
      <button class="u-btn u-btn-primary u-btn-md w-full sm:w-auto">
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
        Добавить датчик
      </button>
    </div>

    <!-- Sensors Grid -->
    <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      <div v-for="sensor in sensors" :key="sensor.id" class="u-card p-6">
        <div class="flex items-start justify-between mb-4">
          <div class="flex-1 min-w-0">
            <h3 class="font-semibold text-gray-900 truncate">{{ sensor.name }}</h3>
            <p class="text-sm text-gray-500">{{ sensor.type }} • {{ sensor.system }}</p>
          </div>
          <span class="u-badge" :class="getSensorStatusClass(sensor.status)">
            <Icon :name="getSensorStatusIcon(sensor.status)" class="w-3 h-3" />
            {{ sensor.status === 'normal' ? 'Норма' : sensor.status === 'warning' ? 'Предупреждение' : 'Критично' }}
          </span>
        </div>
        
        <div class="space-y-3">
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-600">Текущее значение</span>
            <span class="font-semibold">{{ sensor.currentValue }} {{ sensor.unit }}</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-600">Последнее обновление</span>
            <span class="text-sm text-gray-500">{{ sensor.lastUpdate }}</span>
          </div>
        </div>
        
        <div class="flex items-center gap-2 mt-4">
          <button class="u-btn u-btn-ghost u-btn-sm flex-1">
            <Icon name="heroicons:chart-bar" class="w-4 h-4 mr-1" />
            История
          </button>
          <button class="u-btn u-btn-ghost u-btn-sm flex-1">
            <Icon name="heroicons:cog-6-tooth" class="w-4 h-4 mr-1" />
            Настройки
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Page metadata
definePageMeta({
  middleware: ['auth']
})

// Mock sensors data
const sensors = ref([
  {
    id: 1,
    name: 'Датчик давления P001',
    type: 'Давление',
    system: 'HYD-001',
    currentValue: 2.3,
    unit: 'бар',
    status: 'normal' as const,
    lastUpdate: '2 мин назад'
  },
  {
    id: 2,
    name: 'Датчик температуры T001',
    type: 'Температура',
    system: 'HYD-001',
    currentValue: 68,
    unit: '°C',
    status: 'warning' as const,
    lastUpdate: '1 мин назад'
  },
  {
    id: 3,
    name: 'Датчик потока F001',
    type: 'Поток',
    system: 'HYD-002', 
    currentValue: 185,
    unit: 'л/мин',
    status: 'critical' as const,
    lastUpdate: '5 мин назад'
  }
])

// Helper functions with proper typing
const getSensorStatusClass = (status: string): string => {
  const classes: Record<string, string> = {
    normal: 'u-badge-success',
    warning: 'u-badge-warning',
    critical: 'u-badge-error'
  }
  return classes[status] || 'u-badge-info'
}

const getSensorStatusIcon = (status: string): string => {
  const icons: Record<string, string> = {
    normal: 'heroicons:check-circle',
    warning: 'heroicons:exclamation-triangle', 
    critical: 'heroicons:x-circle'
  }
  return icons[status] || 'heroicons:question-mark-circle'
}
</script>