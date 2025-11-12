<template>
  <div class="equipment-overview">
    <BaseCard class="mb-6" title="Базовая информация">
      <div class="flex flex-col gap-2">
        <div><strong>ID:</strong> {{ equipment.id }}</div>
        <div><strong>Название:</strong> {{ equipment.name }}</div>
        <div><strong>Тип:</strong> {{ equipment.equipment_type }}</div>
        <div><strong>Модель:</strong> {{ equipment.model }}</div>
        <div><strong>Производитель:</strong> {{ equipment.manufacturer }}</div>
        <div><strong>Статус:</strong> <StatusBadge :status="mapStatus(equipment.status)" /></div>
        <div><strong>Дата выпуска:</strong> {{ equipment.manufacture_date }}</div>
      </div>
    </BaseCard>
    <BaseCard class="mb-6" title="Архитектура системы">
      <!-- TODO: Интегрировать граф-схему (GraphView.vue) -->
      <div class="text-industrial-500 py-8 text-center">Здесь будет граф визуализации компонентов (GraphView.vue)</div>
    </BaseCard>
    <BaseCard title="Список компонентов">
      <!-- TODO: Таблица по components[] -->
      <div class="text-industrial-500 py-8 text-center">Здесь будет таблица компонентов оборудования</div>
    </BaseCard>
  </div>
</template>
<script setup lang="ts">
defineProps<{ equipment: any }>()
function mapStatus(status: string) {
  switch (status) {
    case 'active': return 'operational'
    case 'inactive': return 'degraded'
    default: return 'unknown'
  }
}
</script>
<style scoped>
.equipment-overview {
  @apply space-y-6;
}
</style>