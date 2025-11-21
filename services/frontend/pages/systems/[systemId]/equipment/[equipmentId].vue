<template>
  <div class="space-y-8">
    <!-- Breadcrumb -->
    <UBreadcrumb :items="breadcrumbs" />

    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ equipment.name }}</h1>
        <p class="text-steel-shine mt-2">{{ equipment.type }}</p>
      </div>
      <div class="flex items-center gap-3">
        <UStatusDot :status="equipment.status" :label="equipment.statusLabel" />
        <UButton variant="secondary" size="md">
          <Icon name="heroicons:cog-6-tooth" class="w-5 h-5 mr-2" />
          Настройки
        </UButton>
      </div>
    </div>

    <!-- Metrics Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <UCard>
        <UCardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-steel-shine">Состояние</p>
              <p class="text-2xl font-bold text-success-400 mt-1">Работает</p>
            </div>
            <div class="w-12 h-12 bg-success-600/20 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:check-circle" class="w-6 h-6 text-success-400" />
            </div>
          </div>
        </UCardContent>
      </UCard>

      <UCard>
        <UCardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-steel-shine">Нагрузка</p>
              <p class="text-2xl font-bold text-primary-400 mt-1">{{ equipment.load }}%</p>
            </div>
            <div class="w-12 h-12 bg-primary-600/20 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:cpu-chip" class="w-6 h-6 text-primary-400" />
            </div>
          </div>
        </UCardContent>
      </UCard>

      <UCard>
        <UCardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-steel-shine">Вибрация</p>
              <p class="text-2xl font-bold text-warning-400 mt-1">{{ equipment.vibration }} mm/s</p>
            </div>
            <div class="w-12 h-12 bg-warning-600/20 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:arrow-trending-up" class="w-6 h-6 text-warning-400" />
            </div>
          </div>
        </UCardContent>
      </UCard>

      <UCard>
        <UCardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium text-steel-shine">Время работы</p>
              <p class="text-2xl font-bold text-info-400 mt-1">{{ equipment.uptime }} ч</p>
            </div>
            <div class="w-12 h-12 bg-info-600/20 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:clock" class="w-6 h-6 text-info-400" />
            </div>
          </div>
        </UCardContent>
      </UCard>
    </div>

    <!-- Details & Events -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Details -->
      <UCard>
        <UCardHeader class="border-b border-steel-700/50">
          <UCardTitle>Подробные данные</UCardTitle>
        </UCardHeader>
        <UCardContent class="space-y-4 p-6">
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">ID оборудования</span>
            <span class="text-sm font-semibold text-white">{{ equipment.id }}</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">Тип</span>
            <span class="text-sm font-semibold text-white">{{ equipment.type }}</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">Статус</span>
            <UBadge variant="success">{{ equipment.statusLabel }}</UBadge>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">Последнее обновление</span>
            <span class="text-xs text-steel-400 flex items-center gap-1">
              <Icon name="heroicons:clock" class="w-3 h-3" />
              {{ formatDate(equipment.lastUpdate) }}
            </span>
          </div>
          <div class="border-t border-steel-700/50 my-4" />
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">Система</span>
            <NuxtLink :to="`/systems/${route.params.systemId}`" class="text-sm font-semibold text-primary-400 hover:text-primary-300">
              Система #{{ route.params.systemId }}
            </NuxtLink>
          </div>
        </UCardContent>
      </UCard>

      <!-- Event Log -->
      <UCard>
        <UCardHeader class="border-b border-steel-700/50">
          <UCardTitle>Журнал событий</UCardTitle>
        </UCardHeader>
        <UCardContent class="space-y-3 p-6">
          <div v-for="event in events" :key="event.id" class="flex items-center gap-3 p-3 rounded-lg" :class="event.bgClass">
            <UStatusDot :status="event.status" />
            <div class="flex-1">
              <p class="text-sm font-medium text-white">{{ event.message }}</p>
              <p class="text-xs text-steel-400">{{ event.time }}</p>
            </div>
          </div>
        </UCardContent>
      </UCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

definePageMeta({
  layout: 'dashboard' as const,
  middleware: ['auth']
})

const route = useRoute()
const systemId = route.params.systemId as string
const equipmentId = route.params.equipmentId as string

const breadcrumbs = computed(() => [
  { label: 'Системы', to: '/systems' },
  { label: `Система #${systemId}`, to: `/systems/${systemId}` },
  { label: 'Оборудование', to: `/systems/${systemId}/equipment` },
  { label: equipmentId, to: '' }
])

interface Equipment {
  id: string
  name: string
  type: string
  status: 'success' | 'warning' | 'error' | 'offline'
  statusLabel: string
  load: number
  vibration: number
  uptime: number
  lastUpdate: string
}

const equipment = ref<Equipment>({
  id: equipmentId,
  name: 'Pump Unit A1',
  type: 'Насосная установка',
  status: 'success',
  statusLabel: 'Работает',
  load: 78,
  vibration: 0.8,
  uptime: 142,
  lastUpdate: new Date().toISOString()
})

const events = ref([
  {
    id: 1,
    status: 'success' as const,
    message: 'Оборудование запущено',
    time: '2 часа назад',
    bgClass: 'bg-success-500/5 border border-success-500/20'
  },
  {
    id: 2,
    status: 'warning' as const,
    message: 'Повышенная вибрация обнаружена',
    time: '1 час назад',
    bgClass: 'bg-warning-500/5 border border-warning-500/20'
  },
  {
    id: 3,
    status: 'info' as const,
    message: 'Плановое техобслуживание завершено',
    time: '5 часов назад',
    bgClass: 'bg-info-500/5 border border-info-500/20'
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
