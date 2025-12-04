<script setup lang="ts">
import { ref, computed } from 'vue';
import type { Component, ComponentType } from '~/types/gnn';
import { ComponentType as ComponentTypeEnum } from '~/types/gnn';
import { useTopologyValidation } from '~/composables/useTopologyValidation';

interface Props {
  modelValue: Component[];
}

const props = defineProps<Props>();
const emit = defineEmits<{
  'update:modelValue': [value: Component[]];
  'validation-change': [isValid: boolean];
}>();

const { validateComponents } = useTopologyValidation();

const components = ref<Component[]>([...props.modelValue]);
const isModalOpen = ref(false);
const editingComponent = ref<Component | null>(null);
const editingIndex = ref<number | null>(null);
const errors = ref<Record<string, string>>({});

// Component Type Options for Select
const componentTypeOptions = Object.entries(ComponentTypeEnum).map(([key, value]) => ({
  label: key.replace(/_/g, ' '),
  value,
}));

// Common sensor options
const sensorOptions = [
  { label: 'Pressure In', value: 'pressure_in' },
  { label: 'Pressure Out', value: 'pressure_out' },
  { label: 'Temperature', value: 'temperature' },
  { label: 'Vibration', value: 'vibration' },
  { label: 'RPM', value: 'rpm' },
  { label: 'Position', value: 'position' },
  { label: 'Flow Rate', value: 'flow_rate' },
];

// Table columns
const columns = [
  { key: 'componentId', label: 'ID', sortable: true },
  { key: 'componentType', label: 'Type', sortable: true },
  { key: 'sensors', label: 'Sensors', render: (val: string[]) => val.join(', ') },
  { key: 'nominalPressureBar', label: 'Pressure (bar)', sortable: true },
  { key: 'nominalFlowLpm', label: 'Flow (L/min)', sortable: true },
  { key: 'actions', label: 'Actions', align: 'right' as const },
];

const validate = () => {
  const validationErrors = validateComponents(components.value);
  
  errors.value = {};
  validationErrors.forEach((err) => {
    errors.value[err.field] = err.message;
  });

  const isValid = validationErrors.length === 0;
  emit('validation-change', isValid);
  emit('update:modelValue', components.value);
  
  return isValid;
};

const isValid = computed(() => {
  return Object.keys(errors.value).length === 0 && components.value.length >= 2;
});

const openAddModal = () => {
  editingComponent.value = {
    componentId: '',
    componentType: ComponentTypeEnum.HYDRAULIC_PUMP,
    sensors: [],
  };
  editingIndex.value = null;
  isModalOpen.value = true;
};

const openEditModal = (component: Component, index: number) => {
  editingComponent.value = { ...component };
  editingIndex.value = index;
  isModalOpen.value = true;
};

const saveComponent = () => {
  if (!editingComponent.value) return;

  if (editingIndex.value !== null) {
    // Update existing
    components.value[editingIndex.value] = { ...editingComponent.value };
  } else {
    // Add new
    components.value.push({ ...editingComponent.value });
  }

  isModalOpen.value = false;
  validate();
};

const deleteComponent = (index: number) => {
  if (confirm('Are you sure you want to delete this component?')) {
    components.value.splice(index, 1);
    validate();
  }
};

// Validate on mount
validate();
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Components Configuration
        </h3>
        <p class="text-sm text-gray-600 dark:text-gray-400">
          Define hydraulic components and their sensors (minimum 2 components required)
        </p>
      </div>
      <Button @click="openAddModal">
        <Icon name="heroicons:plus" class="h-5 w-5 mr-2" />
        Add Component
      </Button>
    </div>

    <!-- Components Table -->
    <Table
      :columns="columns"
      :data="components"
      :empty-text="'No components added yet. Click \"Add Component\" to start.'"
      row-key="componentId"
    >
      <template #cell-actions="{ row, value }">
        <div class="flex items-center justify-end gap-2">
          <Button
            variant="ghost"
            size="sm"
            @click="openEditModal(row, components.findIndex(c => c.componentId === row.componentId))"
          >
            <Icon name="heroicons:pencil" class="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            @click="deleteComponent(components.findIndex(c => c.componentId === row.componentId))"
          >
            <Icon name="heroicons:trash" class="h-4 w-4 text-red-600" />
          </Button>
        </div>
      </template>
    </Table>

    <!-- Validation Summary -->
    <div
      v-if="!isValid"
      class="rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 p-4"
    >
      <div class="flex gap-3">
        <Icon name="heroicons:exclamation-triangle" class="h-5 w-5 text-amber-600 dark:text-amber-400 shrink-0" />
        <div>
          <h4 class="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-1">
            Validation errors:
          </h4>
          <ul class="text-sm text-amber-800 dark:text-amber-200 list-disc list-inside space-y-1">
            <li v-for="(error, field) in errors" :key="field">
              {{ error }}
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Component Edit Modal -->
    <Modal
      v-model="isModalOpen"
      :title="editingIndex !== null ? 'Edit Component' : 'Add Component'"
      size="lg"
    >
      <div v-if="editingComponent" class="space-y-4">
        <!-- Component ID -->
        <Input
          v-model="editingComponent.componentId"
          label="Component ID"
          placeholder="e.g., pump_main_1"
          required
        />

        <!-- Component Type -->
        <Select
          v-model="editingComponent.componentType"
          :options="componentTypeOptions"
          label="Component Type"
          required
        />

        <!-- Sensors Multi-Select -->
        <Select
          v-model="editingComponent.sensors"
          :options="sensorOptions"
          label="Sensors"
          multiple
          searchable
          required
        />

        <!-- Nominal Pressure -->
        <Input
          v-model.number="editingComponent.nominalPressureBar"
          type="number"
          label="Nominal Pressure (bar)"
          placeholder="0-1000"
        />

        <!-- Nominal Flow -->
        <Input
          v-model.number="editingComponent.nominalFlowLpm"
          type="number"
          label="Nominal Flow (L/min)"
          placeholder="0-1000"
        />

        <!-- Rated Power -->
        <Input
          v-model.number="editingComponent.ratedPowerKw"
          type="number"
          label="Rated Power (kW)"
          placeholder="≥0"
        />

        <!-- Metadata (optional) -->
        <div>
          <label class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
            Metadata (optional)
          </label>
          <div class="space-y-2">
            <Input
              v-model="editingComponent.metadata.manufacturer"
              placeholder="Manufacturer"
            />
            <Input
              v-model="editingComponent.metadata.model"
              placeholder="Model"
            />
            <Input
              v-model="editingComponent.metadata.serialNumber"
              placeholder="Serial Number"
            />
          </div>
        </div>
      </div>

      <template #footer="{ close }">
        <div class="flex justify-end gap-3">
          <Button variant="outline" @click="close">
            Cancel
          </Button>
          <Button @click="saveComponent">
            {{ editingIndex !== null ? 'Update' : 'Add' }}
          </Button>
        </div>
      </template>
    </Modal>
  </div>
</template>
