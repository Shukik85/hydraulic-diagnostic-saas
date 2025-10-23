<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Equipment</h1>
        <p class="text-muted-foreground">Manage and monitor your hydraulic systems</p>
      </div>
      <UiButton as-child>
        <NuxtLink to="/equipment/new">
          <Icon name="lucide:plus" class="mr-2 h-4 w-4" />
          Add Equipment
        </NuxtLink>
      </UiButton>
    </div>

    <!-- Filter Toolbar -->
    <UiCard>
      <UiCardContent class="pt-6">
        <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div class="flex flex-1 items-center gap-2">
            <div class="relative flex-1 max-w-sm">
              <Icon name="lucide:search" class="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <UiInput
                placeholder="Search equipment..."
                class="pl-10"
                v-model="searchQuery"
              />
            </div>
            <UiSelect v-model="statusFilter">
              <UiSelectItem value="all">All Status</UiSelectItem>
              <UiSelectItem value="online">Online</UiSelectItem>
              <UiSelectItem value="warning">Warning</UiSelectItem>
              <UiSelectItem value="error">Error</UiSelectItem>
              <UiSelectItem value="offline">Offline</UiSelectItem>
            </UiSelect>
            <UiSelect v-model="typeFilter">
              <UiSelectItem value="all">All Types</UiSelectItem>
              <UiSelectItem value="pump">Pump</UiSelectItem>
              <UiSelectItem value="valve">Valve</UiSelectItem>
              <UiSelectItem value="motor">Motor</UiSelectItem>
              <UiSelectItem value="cylinder">Cylinder</UiSelectItem>
            </UiSelect>
          </div>
          <div class="flex items-center gap-2">
            <UiButton variant="outline" size="sm" @click="viewMode = 'grid'" :class="{ 'bg-muted': viewMode === 'grid' }">
              <Icon name="lucide:grid" class="h-4 w-4" />
            </UiButton>
            <UiButton variant="outline" size="sm" @click="viewMode = 'list'" :class="{ 'bg-muted': viewMode === 'list' }">
              <Icon name="lucide:list" class="h-4 w-4" />
            </UiButton>
          </div>
        </div>
      </UiCardContent>
    </UiCard>

    <!-- Equipment Grid/List -->
    <div v-if="viewMode === 'grid'" class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      <UiCard v-for="equipment in filteredEquipment" :key="equipment.id" class="cursor-pointer hover:shadow-lg transition-shadow" @click="viewEquipment(equipment.id)">
        <UiCardHeader>
          <div class="flex items-center justify-between">
            <UiCardTitle class="text-lg">{{ equipment.name }}</UiCardTitle>
            <div :class="`h-3 w-3 rounded-full ${getStatusColor(equipment.status)}`"></div>
          </div>
          <UiCardDescription>{{ equipment.type }} • {{ equipment.location }}</UiCardDescription>
        </UiCardHeader>
        <UiCardContent>
          <div class="space-y-4">
            <div class="flex items-center justify-between text-sm">
              <span class="text-muted-foreground">Health Score</span>
              <span class="font-medium">{{ equipment.health }}%</span>
            </div>
            <div class="w-full bg-muted rounded-full h-2">
              <div
                class="h-2 rounded-full transition-all duration-300"
                :class="getHealthColor(equipment.health)"
                :style="{ width: equipment.health + '%' }"
              ></div>
            </div>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p class="text-muted-foreground">Last Check</p>
                <p class="font-medium">{{ equipment.lastCheck }}</p>
              </div>
              <div>
                <p class="text-muted-foreground">Next Service</p>
                <p class="font-medium">{{ equipment.nextService }}</p>
              </div>
            </div>
          </div>
        </UiCardContent>
      </UiCard>
    </div>

    <!-- List View -->
    <UiCard v-else>
      <UiCardContent class="p-0">
        <div class="divide-y">
          <div
            v-for="equipment in filteredEquipment"
            :key="equipment.id"
            class="flex items-center justify-between p-4 hover:bg-muted/50 cursor-pointer transition-colors"
            @click="viewEquipment(equipment.id)"
          >
            <div class="flex items-center space-x-4">
              <div :class="`h-3 w-3 rounded-full ${getStatusColor(equipment.status)}`"></div>
              <div>
                <p class="font-medium">{{ equipment.name }}</p>
                <p class="text-sm text-muted-foreground">{{ equipment.type }} • {{ equipment.location }}</p>
              </div>
            </div>
            <div class="flex items-center space-x-6">
              <div class="text-right">
                <p class="text-sm font-medium">{{ equipment.health }}% Health</p>
                <p class="text-xs text-muted-foreground">Last: {{ equipment.lastCheck }}</p>
              </div>
              <Icon name="lucide:chevron-right" class="h-4 w-4 text-muted-foreground" />
            </div>
          </div>
        </div>
      </UiCardContent>
    </UiCard>

    <!-- FAB -->
    <UiButton
      class="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg"
      size="icon"
      as-child
    >
      <NuxtLink to="/equipment/new">
        <Icon name="lucide:plus" class="h-6 w-6" />
      </NuxtLink>
    </UiButton>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const searchQuery = ref('')
const statusFilter = ref('all')
const typeFilter = ref('all')
const viewMode = ref('grid')

const equipment = ref([
  {
    id: '1',
    name: 'HYD-001',
    type: 'Pump Station',
    location: 'Building A',
    status: 'online',
    health: 98,
    lastCheck: '2 hours ago',
    nextService: '2 weeks'
  },
  {
    id: '2',
    name: 'HYD-002',
    type: 'Hydraulic Motor',
    location: 'Building B',
    status: 'warning',
    health: 76,
    lastCheck: '30 min ago',
    nextService: '1 week'
  },
  {
    id: '3',
    name: 'HYD-003',
    type: 'Control Valve',
    location: 'Building A',
    status: 'online',
    health: 95,
    lastCheck: '1 hour ago',
    nextService: '3 weeks'
  },
  {
    id: '4',
    name: 'HYD-004',
    type: 'Cylinder Assembly',
    location: 'Building C',
    status: 'error',
    health: 45,
    lastCheck: '15 min ago',
    nextService: 'Immediate'
  }
])

const filteredEquipment = computed(() => {
  return equipment.value.filter(item => {
    const matchesSearch = item.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
                         item.type.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
                         item.location.toLowerCase().includes(searchQuery.value.toLowerCase())

    const matchesStatus = statusFilter.value === 'all' || item.status === statusFilter.value
    const matchesType = typeFilter.value === 'all' || item.type.toLowerCase().includes(typeFilter.value)

    return matchesSearch && matchesStatus && matchesType
  })
})

const getStatusColor = (status: string) => {
  switch (status) {
    case 'online': return 'bg-status-success'
    case 'warning': return 'bg-status-warning'
    case 'error': return 'bg-status-error'
    case 'offline': return 'bg-muted-foreground'
    default: return 'bg-muted-foreground'
  }
}

const getHealthColor = (health: number) => {
  if (health >= 90) return 'bg-status-success'
  if (health >= 70) return 'bg-status-warning'
  return 'bg-status-error'
}

const viewEquipment = (id: string) => {
  router.push(`/equipment/${id}`)
}
</script>
