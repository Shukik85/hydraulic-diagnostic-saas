<template>
  <div class="equipment-list-page p-6">
    <div class="flex items-center justify-between mb-8 gap-4 flex-wrap">
      <h1 class="text-2xl font-bold">Список оборудования</h1>
      <div class="flex gap-2 items-center flex-wrap">
        <UInput v-model="search" placeholder="Поиск по названию..." class="w-64" icon="heroicons:magnifying-glass" />
        <USelect v-model="typeFilter" :options="equipmentTypes" placeholder="Тип оборудования" class="w-40" />
        <USelect v-model="statusFilter" :options="statusOptions" placeholder="Статус" class="w-32" />
        <BaseButton icon="heroicons:plus" variant="primary" @click="addEquipment">Добавить</BaseButton>
      </div>
    </div>
    <div v-if="filteredEquipment.length === 0" class="p-12 text-center text-industrial-500">
      <Icon name="heroicons:archive-box" class="w-16 h-16 mx-auto mb-3 text-industrial-300" />
      <div class="mb-2">Оборудование не найдено</div>
    </div>
    <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
      <BaseCard v-for="equip in filteredEquipment" :key="equip.id" hover>
        <template #header>
          <div class="flex items-center justify-between">
            <div class="font-semibold">{{ equip.name || equip.model || equip.id }}</div>
            <StatusBadge :status="mapStatus(equip.status)" size="sm"/>
          </div>
        </template>
        <div class="mb-1 text-industrial-500 flex flex-wrap gap-2">
          <span>{{ equip.equipment_type }}</span>
          <span v-if="equip.manufacturer">• {{ equip.manufacturer }}</span>
          <span v-if="equip.model">• {{ equip.model }}</span>
        </div>
        <div class="mt-2 flex gap-2">
          <BaseButton size="sm" @click="viewEquipment(equip.id)">Просмотр</BaseButton>
          <BaseButton size="sm" variant="secondary" @click="editEquipment(equip.id)">Изменить</BaseButton>
          <BaseButton size="sm" variant="danger" @click="deleteEquipment(equip.id)">Удалить</BaseButton>
        </div>
      </BaseCard>
    </div>
  </div>
</template>

<script setup lang="ts">
const search = ref('')
const typeFilter = ref('')
const statusFilter = ref('')
const equipmentList = ref<any[]>([])
const equipmentTypes = [
  { label: 'Все', value: '' },
  { label: 'Экскаватор', value: 'excavator' },
  { label: 'Пресс', value: 'press' },
  { label: 'Кран', value: 'crane' },
  { label: 'Погрузчик', value: 'loader' }
]
const statusOptions = [
  { label: 'Все', value: '' },
  { label: 'Активен', value: 'active' },
  { label: 'Неактивен', value: 'inactive' }
]
// TODO: Загрузить список с бэкенда
onMounted(async () => {
  equipmentList.value = await fetchEquipment()
})
async function fetchEquipment() {
  // Имитация запроса к API
  // Вернуть массив объектов оборудования
  return []
}
const filteredEquipment = computed(() => {
  return equipmentList.value.filter(e =>
    (!search.value || (e.name || e.model || '').toLowerCase().includes(search.value.toLowerCase())) &&
    (!typeFilter.value || e.equipment_type === typeFilter.value) &&
    (!statusFilter.value || e.status === statusFilter.value)
  )
})
function addEquipment() { /* переход к wizard или модалке создания */ }
function viewEquipment(id: string) { navigateTo(`/equipment/${id}`) }
function editEquipment(id: string) { /* переход на страницу редактирования */ }
function deleteEquipment(id: string) { /* вызов API + обновление */ }
function mapStatus(status: string) {
  switch (status) {
    case 'active': return 'operational'
    case 'inactive': return 'degraded'
    default: return 'unknown'
  }
}
</script>

<style scoped>
.equipment-list-page {
  min-height: 100vh;
  background: var(--bg, #f7fafc);
}
</style>