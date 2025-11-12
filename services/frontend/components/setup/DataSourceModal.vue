<template>
  <div class="data-source-modal">
    <UModal v-model="open" :closable="!busy">
      <UCard>
        <template #header>
          <div class="font-semibold text-lg mb-2">Настройка источника данных</div>
        </template>
        <div class="space-y-4">
          <p class="text-industrial-600 dark:text-industrial-400">
            Выберите способ загрузки данных для системы:
          </p>
          <div class="flex gap-4">
            <BaseButton variant="primary" @click="goTo('csv')" :disabled="busy">Upload CSV</BaseButton>
            <BaseButton variant="info" @click="goTo('sim')" :disabled="busy">Use Simulator</BaseButton>
            <BaseButton variant="secondary" @click="skip" :disabled="busy">Skip</BaseButton>
          </div>
        </div>
      </UCard>
    </UModal>
  </div>
</template>
<script setup lang="ts">
const open = ref(true)
const busy = ref(false)
const emit = defineEmits(['go-csv', 'go-sim', 'skip'])
function goTo(type: string) {
  busy.value = true
  if (type === 'csv') emit('go-csv')
  if (type === 'sim') emit('go-sim')
}
function skip() {
  busy.value = true
  emit('skip')
}
</script>