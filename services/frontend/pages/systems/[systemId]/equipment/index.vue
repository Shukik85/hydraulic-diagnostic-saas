<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="u-h2">Оборудование</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          Управление и мониторинг оборудования - Система #{{ route.params.systemId }}
        </p>
      </div>
      <button class="u-btn u-btn-primary u-btn-md" @click="showAddModal = true">
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
        Добавить оборудование
      </button>
    </div>

    <!-- Zero State -->
    <UZeroState
      v-if="!loading && equipment.length === 0"
      icon-name="heroicons:cog-6-tooth"
      title="Нет оборудования"
      description="Добавьте оборудование для мониторинга состояния системы"
      action-icon="heroicons:plus"
      action-text="Добавить оборудование"
      @action="showAddModal = true"
    />

    <!-- Equipment Grid -->
    <div v-else class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      <div
        v-for="item in equipment"
        :key="item.id"
        class="card-interactive p-6"
        role="button"
        tabindex="0"
        @click="navigateTo(`/systems/${route.params.systemId}/equipment/${item.id}`)"
        @keydown.enter="navigateTo(`/systems/${route.params.systemId}/equipment/${item.id}`)"
      >
        <!-- Header -->
        <div class="flex items-start justify-between mb-4">
          <div class="flex-1 min-w-0">
            <h3 class="font-semibold text-white truncate text-lg">{{ item.name }}</h3>
            <p class="text-sm text-steel-shine">{{ item.type }}</p>
          </div>
          <UStatusDot :status="item.status" :label="item.statusLabel" />
        </div>

        <!-- Metrics -->
        <div class="space-y-3 mb-4">
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-steel-400" />
              <span class="text-sm text-steel-shine">Нагрузка</span>
            </div>
            <span class="font-semibold text-white">{{ item.load }}%</span>
          </div>
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:arrow-trending-up" class="w-4 h-4 text-steel-400" />
              <span class="text-sm text-steel-shine">Вибрация</span>
            </div>
            <span class="font-semibold text-white">{{ item.vibration }} mm/s</span>
          </div>
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:clock" class="w-4 h-4 text-steel-400" />
              <span class="text-sm text-steel-shine">Время работы</span>
            </div>
            <span class="font-semibold text-white">{{ item.uptime }} ч</span>
          </div>
        </div>

        <!-- Footer -->
        <div class="flex items-center justify-between pt-4 border-t border-steel-700/50">
          <div class="flex items-center gap-1.5 text-xs text-steel-400">
            <Icon name="heroicons:clock" class="w-3 h-3" />
            <span>Обновлено {{ formatDate(item.lastUpdate) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

definePageMeta({
  layout: 'dashboard' as const,
  middleware: ['auth']
})

const route = useRoute()
const showAddModal = ref(false)
const loading = ref(false)

interface Equipment {
  id: number
  name: string
  type: string
  status: 'success' | 'warning' | 'error' | 'offline'
  statusLabel: string
  load: number
  vibration: number
  uptime: number
  lastUpdate: string
}

const equipment = ref<Equipment[]>([
  {
    id: 1,
    name: 'Pump Unit A1',
    type: 'Насосная установка',
    status: 'success',
    statusLabel: 'Работает',
    load: 78,
    vibration: 0.8,
    uptime: 142,
    lastUpdate: new Date().toISOString()
  },
  {
    id: 2,
    name: 'Valve Control V2',
    type: 'Управляющий клапан',
    status: 'warning',
    statusLabel: 'Предупреждение',
    load: 92,
    vibration: 1.5,
    uptime: 87,
    lastUpdate: new Date().toISOString()
  },
  {
    id: 3,
    name: 'Motor Drive M3',
    type: 'Гидромотор',
    status: 'success',
    statusLabel: 'Работает',
    load: 65,
    vibration: 0.6,
    uptime: 210,
    lastUpdate: new Date().toISOString()
  }
])

const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}
</script>
