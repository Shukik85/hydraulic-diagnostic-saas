<template>
  <div class="w-full">
    <!-- Desktop Table -->
    <div class="hidden md:block premium-card overflow-hidden">
      <table class="w-full">
        <thead class="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <tr>
            <th
              v-for="column in columns"
              :key="column.key"
              class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              <div class="flex items-center space-x-1">
                <span>{{ column.label }}</span>
                <button
                  v-if="column.sortable"
                  @click="toggleSort(column.key)"
                  class="ml-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <Icon name="heroicons:chevron-up-down" class="w-4 h-4" />
                </button>
              </div>
            </th>
            <th v-if="$slots.actions" class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
          <tr
            v-for="(item, index) in sortedData"
            :key="getItemKey(item, index)"
            class="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            <td
              v-for="column in columns"
              :key="column.key"
              class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white"
            >
              <slot :name="column.key" :item="item" :value="getItemValue(item, column.key)">
                {{ getItemValue(item, column.key) }}
              </slot>
            </td>
            <td v-if="$slots.actions" class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
              <slot name="actions" :item="item" :index="index" />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    
    <!-- Mobile Cards -->
    <div class="md:hidden space-y-4">
      <div
        v-for="(item, index) in sortedData"
        :key="getItemKey(item, index)"
        class="premium-card"
      >
        <div class="p-4 border-b border-gray-200 dark:border-gray-700">
          <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            <slot name="card-title" :item="item">
              {{ getItemValue(item, columns?.[0]?.key || 'id') || 'Item' }}
            </slot>
          </h3>
        </div>
        <div class="p-4 space-y-3">
          <div
            v-for="column in columns.slice(1)"
            :key="column.key"
            class="flex justify-between items-center"
          >
            <span class="text-sm font-medium text-gray-500 dark:text-gray-400">{{ column.label }}</span>
            <span class="text-sm text-gray-900 dark:text-white">
              <slot :name="column.key" :item="item" :value="getItemValue(item, column.key)">
                {{ getItemValue(item, column.key) }}
              </slot>
            </span>
          </div>
          <div v-if="$slots.actions" class="flex justify-end pt-2 border-t border-gray-200 dark:border-gray-700">
            <slot name="actions" :item="item" :index="index" />
          </div>
        </div>
      </div>
    </div>
    
    <!-- Empty State -->
    <div v-if="!data || data.length === 0" class="premium-card p-12 text-center">
      <Icon name="heroicons:table-cells" class="w-12 h-12 mx-auto text-gray-400 mb-4" />
      <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No data available</h3>
      <p class="text-gray-500 dark:text-gray-400">There are no items to display in this table.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { TableColumn } from '~/types/api'

interface TableProps {
  data?: any[]
  columns: TableColumn[]
  sortBy?: string
  sortDirection?: 'asc' | 'desc'
}

interface TableEmits {
  'update:sortBy': [key: string]
  'update:sortDirection': [direction: 'asc' | 'desc']
}

const props = withDefaults(defineProps<TableProps>(), {
  data: () => [],
  sortDirection: 'asc'
})

const emit = defineEmits<TableEmits>()

// Safe item key extraction
const getItemKey = (item: any, index: number): string | number => {
  if (item && typeof item === 'object') {
    return item.id || item.key || index
  }
  return index
}

// Safe item value extraction with null checks
const getItemValue = (item: any, key: string): any => {
  if (!item || !key) return ''
  
  // Handle nested keys (e.g., 'user.name')
  const keys = key.split('.')
  let value = item
  
  for (const k of keys) {
    if (value && typeof value === 'object' && k in value) {
      value = value[k]
    } else {
      return ''
    }
  }
  
  return value ?? ''
}

// Sorting functionality
const toggleSort = (key: string) => {
  if (props.sortBy === key) {
    emit('update:sortDirection', props.sortDirection === 'asc' ? 'desc' : 'asc')
  } else {
    emit('update:sortBy', key)
    emit('update:sortDirection', 'asc')
  }
}

const sortedData = computed(() => {
  if (!props.data || !props.sortBy) return props.data || []
  
  return [...props.data].sort((a, b) => {
    const aValue = getItemValue(a, props.sortBy!)
    const bValue = getItemValue(b, props.sortBy!)
    
    // Handle different data types
    let comparison = 0
    
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      comparison = aValue - bValue
    } else {
      const aStr = String(aValue || '').toLowerCase()
      const bStr = String(bValue || '').toLowerCase()
      comparison = aStr.localeCompare(bStr)
    }
    
    return props.sortDirection === 'desc' ? -comparison : comparison
  })
})
</script>