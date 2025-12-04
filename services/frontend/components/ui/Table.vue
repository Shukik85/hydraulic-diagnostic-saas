<script setup lang="ts" generic="T extends Record<string, any>">
import { ref, computed } from 'vue';

interface TableColumn<T> {
  key: keyof T;
  label: string;
  sortable?: boolean;
  width?: string;
  align?: 'left' | 'center' | 'right';
  render?: (value: T[keyof T], row: T) => string;
}

interface TableProps<T> {
  columns: TableColumn<T>[];
  data: T[];
  loading?: boolean;
  selectable?: boolean;
  multiSelect?: boolean;
  emptyText?: string;
  rowKey?: keyof T;
}

const props = withDefaults(defineProps<TableProps<T>>(), {
  loading: false,
  selectable: false,
  multiSelect: false,
  emptyText: 'No data available',
  rowKey: 'id' as keyof T,
});

const emit = defineEmits<{
  'row-click': [row: T];
  'selection-change': [selected: T[]];
}>;

const sortColumn = ref<keyof T | null>(null);
const sortDirection = ref<'asc' | 'desc'>('asc');
const selectedRows = ref<Set<any>>(new Set());

const sortedData = computed(() => {
  if (!sortColumn.value) return props.data;

  return [...props.data].sort((a, b) => {
    const aVal = a[sortColumn.value!];
    const bVal = b[sortColumn.value!];

    if (aVal === bVal) return 0;

    const comparison = aVal > bVal ? 1 : -1;
    return sortDirection.value === 'asc' ? comparison : -comparison;
  });
});

const allSelected = computed(() => {
  if (props.data.length === 0) return false;
  return props.data.every(row => selectedRows.value.has(row[props.rowKey]));
});

const someSelected = computed(() => {
  return selectedRows.value.size > 0 && !allSelected.value;
});

const handleSort = (column: TableColumn<T>) => {
  if (!column.sortable) return;

  if (sortColumn.value === column.key) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc';
  } else {
    sortColumn.value = column.key;
    sortDirection.value = 'asc';
  }
};

const handleRowClick = (row: T) => {
  emit('row-click', row);
};

const toggleRow = (row: T) => {
  const key = row[props.rowKey];
  
  if (props.multiSelect) {
    if (selectedRows.value.has(key)) {
      selectedRows.value.delete(key);
    } else {
      selectedRows.value.add(key);
    }
  } else {
    selectedRows.value.clear();
    selectedRows.value.add(key);
  }

  emitSelectionChange();
};

const toggleAll = () => {
  if (allSelected.value) {
    selectedRows.value.clear();
  } else {
    props.data.forEach(row => {
      selectedRows.value.add(row[props.rowKey]);
    });
  }

  emitSelectionChange();
};

const isRowSelected = (row: T) => {
  return selectedRows.value.has(row[props.rowKey]);
};

const emitSelectionChange = () => {
  const selected = props.data.filter(row => selectedRows.value.has(row[props.rowKey]));
  emit('selection-change', selected);
};

const getCellValue = (row: T, column: TableColumn<T>) => {
  const value = row[column.key];
  return column.render ? column.render(value, row) : String(value);
};

const getAlignClass = (align?: 'left' | 'center' | 'right') => {
  const classes = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right',
  };
  return classes[align || 'left'];
};
</script>

<template>
  <div class="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700" role="table">
      <!-- Header -->
      <thead class="bg-gray-50 dark:bg-gray-800">
        <tr>
          <!-- Selection column -->
          <th
            v-if="selectable"
            scope="col"
            class="w-12 px-4 py-3"
          >
            <Checkbox
              v-if="multiSelect"
              :model-value="allSelected"
              :indeterminate="someSelected"
              @update:model-value="toggleAll"
            />
          </th>

          <!-- Data columns -->
          <th
            v-for="column in columns"
            :key="String(column.key)"
            scope="col"
            :style="column.width ? { width: column.width } : {}"
            :class="[
              'px-4 py-3 text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider',
              getAlignClass(column.align),
              column.sortable ? 'cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-700' : '',
            ]"
            @click="handleSort(column)"
          >
            <div class="flex items-center gap-2" :class="getAlignClass(column.align)">
              <span>{{ column.label }}</span>
              <Icon
                v-if="column.sortable"
                :name="
                  sortColumn === column.key
                    ? sortDirection === 'asc'
                      ? 'heroicons:chevron-up'
                      : 'heroicons:chevron-down'
                    : 'heroicons:chevron-up-down'
                "
                class="h-4 w-4"
                :class="sortColumn === column.key ? 'text-primary-600' : 'text-gray-400'"
                aria-hidden="true"
              />
            </div>
          </th>
        </tr>
      </thead>

      <!-- Body -->
      <tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
        <!-- Loading state -->
        <tr v-if="loading">
          <td
            :colspan="columns.length + (selectable ? 1 : 0)"
            class="px-4 py-8 text-center text-gray-500 dark:text-gray-400"
          >
            <div class="flex items-center justify-center gap-2">
              <Icon name="heroicons:arrow-path" class="h-5 w-5 animate-spin" />
              <span>Loading...</span>
            </div>
          </td>
        </tr>

        <!-- Empty state -->
        <tr v-else-if="data.length === 0">
          <td
            :colspan="columns.length + (selectable ? 1 : 0)"
            class="px-4 py-8 text-center text-gray-500 dark:text-gray-400"
          >
            {{ emptyText }}
          </td>
        </tr>

        <!-- Data rows -->
        <tr
          v-else
          v-for="row in sortedData"
          :key="String(row[rowKey])"
          class="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          :class="[
            isRowSelected(row) ? 'bg-primary-50 dark:bg-primary-900/20' : '',
            'cursor-pointer',
          ]"
          @click="handleRowClick(row)"
        >
          <!-- Selection cell -->
          <td v-if="selectable" class="px-4 py-3" @click.stop>
            <Checkbox
              :model-value="isRowSelected(row)"
              @update:model-value="toggleRow(row)"
            />
          </td>

          <!-- Data cells -->
          <td
            v-for="column in columns"
            :key="String(column.key)"
            class="px-4 py-3 text-sm text-gray-900 dark:text-gray-100"
            :class="getAlignClass(column.align)"
          >
            <slot :name="`cell-${String(column.key)}`" :row="row" :value="row[column.key]">
              {{ getCellValue(row, column) }}
            </slot>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
