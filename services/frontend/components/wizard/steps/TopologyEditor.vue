<script setup lang="ts">
import { ref, computed } from 'vue';
import type { Edge, Component, EdgeType, EdgeMaterial, FlowDirection } from '~/types/gnn';
import { EdgeType as EdgeTypeEnum, EdgeMaterial as EdgeMaterialEnum } from '~/types/gnn';
import { useTopologyValidation } from '~/composables/useTopologyValidation';

interface Props {
  modelValue: Edge[];
  components: Component[];
}

const props = defineProps<Props>();
const emit = defineEmits<{
  'update:modelValue': [value: Edge[]];
  'validation-change': [isValid: boolean];
}>();

const { validateEdges } = useTopologyValidation();

const edges = ref<Edge[]>([...props.modelValue]);
const isModalOpen = ref(false);
const editingEdge = ref<Edge | null>(null);
const editingIndex = ref<number | null>(null);
const errors = ref<Record<string, string>>({});

// Component options for Select (source/target)
const componentOptions = computed(() =>
  props.components.map((c) => ({
    label: `${c.componentId} (${c.componentType})`,
    value: c.componentId,
  }))
);

// Edge Type options
const edgeTypeOptions = Object.entries(EdgeTypeEnum).map(([key, value]) => ({
  label: key.replace(/_/g, ' '),
  value,
}));

// Edge Material options
const edgeMaterialOptions = Object.entries(EdgeMaterialEnum).map(([key, value]) => ({
  label: key.charAt(0) + key.slice(1).toLowerCase(),
  value,
}));

// Flow Direction options
const flowDirectionOptions = [
  { label: 'Unidirectional', value: 'unidirectional' as FlowDirection },
  { label: 'Bidirectional', value: 'bidirectional' as FlowDirection },
];

// Table columns
const columns = [
  { key: 'sourceId', label: 'Source', sortable: true },
  { key: 'targetId', label: 'Target', sortable: true },
  { key: 'edgeType', label: 'Type', sortable: true },
  { key: 'diameterMm', label: 'Diameter (mm)', sortable: true },
  { key: 'lengthM', label: 'Length (m)', sortable: true },
  { key: 'flowDirection', label: 'Flow', sortable: true },
  { key: 'actions', label: 'Actions', align: 'right' as const },
];

const validate = () => {
  const validationErrors = validateEdges(edges.value, props.components);
  
  errors.value = {};
  validationErrors.forEach((err) => {
    errors.value[err.field] = err.message;
  });

  const isValid = validationErrors.length === 0;
  emit('validation-change', isValid);
  emit('update:modelValue', edges.value);
  
  return isValid;
};

const isValid = computed(() => {
  return Object.keys(errors.value).length === 0 && edges.value.length >= 1;
});

const openAddModal = () => {
  editingEdge.value = {
    sourceId: '',
    targetId: '',
    edgeType: EdgeTypeEnum.HYDRAULIC_LINE,
    flowDirection: 'unidirectional',
  };
  editingIndex.value = null;
  isModalOpen.value = true;
};

const openEditModal = (edge: Edge, index: number) => {
  editingEdge.value = { ...edge };
  editingIndex.value = index;
  isModalOpen.value = true;
};

const saveEdge = () => {
  if (!editingEdge.value) return;

  if (editingIndex.value !== null) {
    edges.value[editingIndex.value] = { ...editingEdge.value };
  } else {
    edges.value.push({ ...editingEdge.value });
  }

  isModalOpen.value = false;
  validate();
};

const deleteEdge = (index: number) => {
  if (confirm('Are you sure you want to delete this connection?')) {
    edges.value.splice(index, 1);
    validate();
  }
};

validate();
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Topology Connections
        </h3>
        <p class="text-sm text-gray-600 dark:text-gray-400">
          Define connections (edges) between components (minimum 1 edge required)
        </p>
      </div>
      <Button @click="openAddModal" :disabled="props.components.length < 2">
        <Icon name="heroicons:plus" class="h-5 w-5 mr-2" />
        Add Connection
      </Button>
    </div>

    <!-- Info alert if not enough components -->
    <div
      v-if="props.components.length < 2"
      class="rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 p-4"
    >
      <div class="flex gap-3">
        <Icon name="heroicons:information-circle" class="h-5 w-5 text-blue-600 dark:text-blue-400 shrink-0" />
        <p class="text-sm text-blue-900 dark:text-blue-100">
          Add at least 2 components in Step 2 before defining connections.
        </p>
      </div>
    </div>

    <!-- Edges Table -->
    <Table
      :columns="columns"
      :data="edges"
      :empty-text="'No connections added yet. Click \"Add Connection\" to start.'"
      row-key="sourceId"
    >
      <template #cell-actions="{ row, value }">
        <div class="flex items-center justify-end gap-2">
          <Button
            variant="ghost"
            size="sm"
            @click="openEditModal(row, edges.findIndex(e => e.sourceId === row.sourceId && e.targetId === row.targetId))"
          >
            <Icon name="heroicons:pencil" class="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            @click="deleteEdge(edges.findIndex(e => e.sourceId === row.sourceId && e.targetId === row.targetId))"
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

    <!-- Edge Edit Modal -->
    <Modal
      v-model="isModalOpen"
      :title="editingIndex !== null ? 'Edit Connection' : 'Add Connection'"
      size="lg"
    >
      <div v-if="editingEdge" class="space-y-4">
        <!-- Source Component -->
        <Select
          v-model="editingEdge.sourceId"
          :options="componentOptions"
          label="Source Component"
          searchable
          required
        />

        <!-- Target Component -->
        <Select
          v-model="editingEdge.targetId"
          :options="componentOptions"
          label="Target Component"
          searchable
          required
        />

        <!-- Edge Type -->
        <Select
          v-model="editingEdge.edgeType"
          :options="edgeTypeOptions"
          label="Connection Type"
          required
        />

        <!-- Diameter -->
        <Input
          v-model.number="editingEdge.diameterMm"
          type="number"
          label="Diameter (mm)"
          placeholder="0-500"
        />

        <!-- Length -->
        <Input
          v-model.number="editingEdge.lengthM"
          type="number"
          label="Length (m)"
          placeholder="0-1000"
        />

        <!-- Pressure Rating -->
        <Input
          v-model.number="editingEdge.pressureRatingBar"
          type="number"
          label="Pressure Rating (bar)"
          placeholder="0-1000"
        />

        <!-- Material -->
        <Select
          v-model="editingEdge.material"
          :options="edgeMaterialOptions"
          label="Material"
        />

        <!-- Flow Direction -->
        <Select
          v-model="editingEdge.flowDirection"
          :options="flowDirectionOptions"
          label="Flow Direction"
          required
        />

        <!-- Quick Disconnect -->
        <Checkbox
          v-model="editingEdge.hasQuickDisconnect"
          label="Has Quick Disconnect"
        />
      </div>

      <template #footer="{ close }">
        <div class="flex justify-end gap-3">
          <Button variant="outline" @click="close">
            Cancel
          </Button>
          <Button @click="saveEdge">
            {{ editingIndex !== null ? 'Update' : 'Add' }}
          </Button>
        </div>
      </template>
    </Modal>
  </div>
</template>
