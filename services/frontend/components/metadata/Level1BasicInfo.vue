<script setup lang="ts">
import { ref } from '#imports'

const formData = ref<{
  equipment_type?: string
  name?: string
}>({})

function generateId(): string {
  if (!formData.value.equipment_type) {
    return `XX-${Date.now()}`
  }
  
  // Enterprise: безопасный split с дефолтом
  const parts = formData.value.equipment_type.split('_')
  const firstPart = parts[0] ?? 'XX'
  const prefix = firstPart.toUpperCase().slice(0, 2) || 'XX'
  
  return `${prefix}-${Date.now()}`
}
</script>

<template>
  <div class="basic-info">
    <!-- Form fields -->
    <button @click="generateId">Сгенерировать ID</button>
  </div>
</template>
