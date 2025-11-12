<template>
  <div class="equipment-settings-tab">
    <BaseCard title="Настройки оборудования">
      <!-- TODO: Форма редактирования, удаление -->
      <div class="text-industrial-500 py-8 text-center">Здесь будет форма редактирования и опции удаления оборудования</div>
    </BaseCard>
  </div>
</template>
<script setup lang="ts">
defineProps<{ equipment: any }>()
</script>