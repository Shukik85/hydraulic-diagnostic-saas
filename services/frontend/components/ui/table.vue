<template>
  <div class="w-full">
    <!-- Desktop Table -->
    <div class="hidden md:block card-metal overflow-hidden">
      <table class="w-full">
        <thead class="bg-steel-darker border-b border-steel-medium">
          <tr>
            <th
              v-for="column in columns"
              :key="column.key"
              class="px-6 py-3 text-left text-xs font-bold text-text-secondary uppercase tracking-wider"
            >
              <div class="flex items-center space-x-1">
                <span>{{ column.label }}</span>
                <button
                  v-if="column.sortable"
                  @click="toggleSort(column.key)"
                  class="ml-2 text-steel-light hover:text-primary-400 transition-colors"
                >
                  <Icon name="heroicons:chevron-up-down" class="w-4 h-4" />
                </button>
              </div>
            </th>
            <th
              v-if="$slots.actions"
              class="px-6 py-3 text-right text-xs font-bold text-text-secondary uppercase tracking-wider"
            >
              Actions
            </th>
          </tr>
        </thead>
        <tbody class="bg-steel-dark divide-y divide-steel-medium">
          <tr
            v-for="(item, index) in sortedData"
            :key="getItemKey(item, index)"
            class="hover:bg-primary-600/5 transition-colors duration-200"
          >
            <td
              v-for="column in columns"
              :key="column.key"
              class="px-6 py-4 whitespace-nowrap text-sm text-text-primary"
            >
              <slot :name="column.key" :item="item" :value="getItemValue(item, column.key)">
                {{ getItemValue(item, column.key) }}
              </slot>
            </td>
            <td
              v-if="$slots.actions"
              class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium"
            >
              <slot name="actions" :item="item" :index="index" />
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Mobile Cards -->
    <div class="md:hidden space-y-4">
      <div v-for="(item, index) in sortedData" :key="getItemKey(item, index)" class="card-metal">
        <div class="p-4 border-b border-steel-medium bg-steel-darker">
          <h3 class="text-base font-semibold text-text-primary">
            <slot name="card-title" :item="item">
              {{ getItemValue(item, columns?.[0]?.key || 'id') || 'Item' }}
            </slot>
          </h3>
        </div>
        <div class="p-4 space-y-3 bg-steel-dark">
          <div
            v-for="column in columns.slice(1)"
            :key="column.key"
            class="flex justify-between items-center"
          >
            <span class="text-sm font-medium text-text-secondary">{{ column.label }}</span>
            <span class="text-sm text-text-primary">
              <slot :name="column.key" :item="item" :value="getItemValue(item, column.key)">
                {{ getItemValue(item, column.key) }}
              </slot>
            </span>
          </div>
          <div
            v-if="$slots.actions"
            class="flex justify-end pt-2 border-t border-steel-medium"
          >
            <slot name="actions" :item="item" :index="index" />
          </div>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!data || data.length === 0" class="card-metal p-12 text-center">
      <Icon name="heroicons:table-cells" class="w-12 h-12 mx-auto text-steel-light mb-4" />
      <h3 class="text-lg font-semibold text-text-primary mb-2">No data available</h3>
      <p class="text-text-secondary">There are no items to display in this table.</p>
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
  sortDirection: 'asc',
})

const emit = defineEmits<TableEmits>()

const getItemKey = (item: any, index: number): string | number => {
  if (item && typeof item === 'object') {
    return item.id || item.key || index
  }
  return index
}

const getItemValue = (item: any, key: string): any => {
  if (!item || !key) return ''
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