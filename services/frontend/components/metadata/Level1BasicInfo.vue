<script setup lang="ts">
import { ref } from '#imports'

const formData = ref<{
  equipment_type?: string
  name?: string
}>({})

function generateId() {
  if (formData.value.equipment_type) {
    const parts = formData.value.equipment_type.split('_')
    const prefix = parts[0] ? parts[0].toUpperCase().slice(0, 2) : 'XX'
    return `${prefix}-${Date.now()}`
  }
  return `XX-${Date.now()}`
}
</script>

<template>
  <div class="basic-info">
    <!-- Form fields -->
  </div>
</template>
