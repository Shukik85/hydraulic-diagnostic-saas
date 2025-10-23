<template>
  <div class="w-full">
    <!-- Table Header with Search and Filters -->
    <div class="flex flex-col sm:flex-row gap-4 mb-4">
      <div class="flex-1">
        <UiInput
          v-model="searchQuery"
          placeholder="Поиск..."
          class="max-w-sm"
        />
      </div>
      <div class="flex gap-2">
        <UiSelect v-model="itemsPerPage" class="w-20">
          <UiSelectItem value="5">5</UiSelectItem>
          <UiSelectItem value="10">10</UiSelectItem>
          <UiSelectItem value="25">25</UiSelectItem>
          <UiSelectItem value="50">50</UiSelectItem>
        </UiSelect>
        <UiButton
          variant="outline"
          size="sm"
          @click="toggleView"
          class="hidden sm:flex"
        >
          <Icon name="lucide:grid" class="h-4 w-4" v-if="viewMode === 'table'" />
          <Icon name="lucide:list" class="h-4 w-4" v-else />
        </UiButton>
      </div>
    </div>

    <!-- Table View -->
    <div v-if="viewMode === 'table'" class="rounded-md border">
      <div class="overflow-x-auto">
        <table class="w-full caption-bottom text-sm">
          <thead class="border-b bg-muted/50">
            <tr>
              <th
                v-for="column in columns"
                :key="column.key"
                class="h-12 px-4 text-left align-middle font-medium text-muted-foreground cursor-pointer hover:bg-muted/70 transition-colors"
                @click="sortBy(column.key)"
              >
                <div class="flex items-center gap-2">
                  {{ column.label }}
                  <Icon
                    :name="sortKey === column.key && sortOrder === 'asc' ? 'lucide:arrow-up' : 'lucide:arrow-down'"
                    class="h-4 w-4"
                    :class="{ 'opacity-50': sortKey !== column.key }"
                  />
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="item in paginatedItems"
              :key="item.id"
              class="border-b hover:bg-muted/50 transition-colors"
            >
              <td
                v-for="column in columns"
                :key="column.key"
                class="p-4 align-middle"
              >
                <slot :name="`column-${column.key}`" :item="item" :value="item[column.key]">
                  {{ item[column.key] }}
                </slot>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Card View for Mobile -->
    <div v-else class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <UiCard
        v-for="item in paginatedItems"
        :key="item.id"
        class="cursor-pointer hover:shadow-md transition-shadow"
      >
        <UiCardHeader>
          <UiCardTitle class="text-lg">
            <slot name="card-title" :item="item">
              {{ item[columns[0]?.key] || 'Item' }}
            </slot>
          </UiCardTitle>
        </UiCardHeader>
        <UiCardContent>
          <div class="space-y-2">
            <div
              v-for="column in columns.slice(1)"
              :key="column.key"
              class="flex justify-between items-center"
            >
              <span class="text-sm text-muted-foreground">{{ column.label }}:</span>
              <span class="text-sm font-medium">
                <slot :name="`column-${column.key}`" :item="item" :value="item[column.key]">
                  {{ item[column.key] }}
                </slot>
              </span>
            </div>
          </div>
        </UiCardContent>
      </UiCard>
    </div>

    <!-- Pagination -->
    <div class="flex items-center justify-between mt-4">
      <div class="text-sm text-muted-foreground">
        Показано {{ startIndex + 1 }}-{{ Math.min(endIndex, filteredItems.length) }} из {{ filteredItems.length }}
      </div>
      <div class="flex gap-2">
        <UiButton
          variant="outline"
          size="sm"
          :disabled="currentPage === 1"
          @click="currentPage--"
        >
          <Icon name="lucide:chevron-left" class="h-4 w-4" />
          Предыдущая
        </UiButton>
        <UiButton
          variant="outline"
          size="sm"
          :disabled="currentPage === totalPages"
          @click="currentPage++"
        >
          Следующая
          <Icon name="lucide:chevron-right" class="h-4 w-4" />
        </UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

interface Column {
  key: string
  label: string
  sortable?: boolean
}

interface Props {
  columns: Column[]
  items: any[]
  searchable?: boolean
  sortable?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  searchable: true,
  sortable: true,
})

const searchQuery = ref('')
const sortKey = ref('')
const sortOrder = ref<'asc' | 'desc'>('asc')
const currentPage = ref(1)
const itemsPerPage = ref(10)
const viewMode = ref<'table' | 'card'>('table')

const filteredItems = computed(() => {
  let filtered = props.items

  if (searchQuery.value) {
    filtered = filtered.filter(item =>
      Object.values(item).some(value =>
        String(value).toLowerCase().includes(searchQuery.value.toLowerCase())
      )
    )
  }

  if (sortKey.value) {
    filtered = [...filtered].sort((a, b) => {
      const aVal = a[sortKey.value]
      const bVal = b[sortKey.value]

      if (aVal < bVal) return sortOrder.value === 'asc' ? -1 : 1
      if (aVal > bVal) return sortOrder.value === 'asc' ? 1 : -1
      return 0
    })
  }

  return filtered
})

const totalPages = computed(() => Math.ceil(filteredItems.value.length / itemsPerPage.value))
const startIndex = computed(() => (currentPage.value - 1) * itemsPerPage.value)
const endIndex = computed(() => startIndex.value + itemsPerPage.value)

const paginatedItems = computed(() =>
  filteredItems.value.slice(startIndex.value, endIndex.value)
)

const sortBy = (key: string) => {
  if (!props.sortable) return

  if (sortKey.value === key) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    sortOrder.value = 'asc'
  }
}

const toggleView = () => {
  viewMode.value = viewMode.value === 'table' ? 'card' : 'table'
}

watch(filteredItems, () => {
  currentPage.value = 1
})

watch(itemsPerPage, () => {
  currentPage.value = 1
})
</script>
