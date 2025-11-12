<template>
  <div class="equipment-sensors-tab">
    <BaseCard class="mb-6" title="Привязка датчиков">
      <!-- TODO: Таблица sensor mappings, добавление/редактирование -->
      <div class="text-industrial-500 py-8 text-center">Здесь будет таблица и настройки привязок датчиков</div>
    </BaseCard>
    <BaseCard title="Текущие сенсоры">
      <!-- TODO: Информация о текущих активных сенсорах -->
      <div class="text-industrial-500 py-8 text-center">Список сенсоров по оборудованию</div>
    </BaseCard>
  </div>
</template>
<script setup lang="ts">
defineProps<{ equipment: any }>()
</script>
<style scoped>
.equipment-sensors-tab {
  @apply space-y-6;
}
</style>