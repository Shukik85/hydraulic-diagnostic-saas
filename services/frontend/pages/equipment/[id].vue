<template>
  <div class="equipment-detail-page p-6">
    <h1 class="text-2xl font-bold mb-6">Детали оборудования</h1>
    <UTabs v-model="tab" :items="tabItems" class="mb-8" />
    <div v-if="tab === 0"><EquipmentOverview :equipment="equipment" /></div>
    <div v-else-if="tab === 1"><EquipmentSensors :equipment="equipment" /></div>
    <div v-else-if="tab === 2"><EquipmentDataSources :equipment="equipment" /></div>
    <div v-else-if="tab === 3"><EquipmentSettings :equipment="equipment" /></div>
  </div>
</template>

<script setup lang="ts">
import EquipmentOverview from '~/components/equipment/EquipmentOverview.vue'
import EquipmentSensors from '~/components/equipment/EquipmentSensors.vue'
import EquipmentDataSources from '~/components/equipment/EquipmentDataSources.vue'
import EquipmentSettings from '~/components/equipment/EquipmentSettings.vue'

const route = useRoute()
const equipmentId = computed(() => route.params.id)
const equipment = ref<any>(null)
const tab = ref(0)
const tabItems = [
  { label: 'Обзор' },
  { label: 'Датчики' },
  { label: 'Источники данных' },
  { label: 'Настройки' }
]

onMounted(async () => {
  equipment.value = await fetchEquipmentDetail(equipmentId.value)
})
async function fetchEquipmentDetail(id: string) {
  // Имитация запроса — заменить на реальный fetch с API
  return { id, name: 'Оборудование', model: 'Модель', equipment_type: '', manufacturer: '', status: '' }
}
</script>

<style scoped>
.equipment-detail-page {
  min-height: 100vh;
  background: var(--bg, #f7fafc);
}
</style>