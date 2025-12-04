<script setup lang="ts">
import { ref, computed } from 'vue';
import type { Component } from '~/types/gnn';

interface Props {
  modelValue: Component[];
}

const props = defineProps<Props>();
const emit = defineEmits<{
  'update:modelValue': [value: Component[]];
  'validation-change': [isValid: boolean];
}>;

const fileInputRef = ref<HTMLInputElement | null>(null);
const uploadedFile = ref<File | null>(null);
const parsedData = ref<Component[]>([]);
const parseError = ref<string | null>(null);
const isDragging = ref(false);
const isProcessing = ref(false);

const acceptedFormats = '.csv,.json,.xlsx,.xls';

const fileInfo = computed(() => {
  if (!uploadedFile.value) return null;
  
  return {
    name: uploadedFile.value.name,
    size: (uploadedFile.value.size / 1024).toFixed(2) + ' KB',
    type: uploadedFile.value.type,
  };
});

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    processFile(target.files[0]);
  }
};

const handleDrop = (event: DragEvent) => {
  isDragging.value = false;
  
  if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
    processFile(event.dataTransfer.files[0]);
  }
};

const handleDragOver = (event: DragEvent) => {
  event.preventDefault();
  isDragging.value = true;
};

const handleDragLeave = () => {
  isDragging.value = false;
};

const openFilePicker = () => {
  fileInputRef.value?.click();
};

const processFile = async (file: File) => {
  uploadedFile.value = file;
  parseError.value = null;
  isProcessing.value = true;

  try {
    const extension = file.name.split('.').pop()?.toLowerCase();

    if (extension === 'json') {
      await parseJSON(file);
    } else if (extension === 'csv') {
      await parseCSV(file);
    } else if (extension === 'xlsx' || extension === 'xls') {
      parseError.value = 'Excel files not yet supported. Please use CSV or JSON.';
    } else {
      parseError.value = 'Unsupported file format. Please use CSV or JSON.';
    }
  } catch (error: any) {
    parseError.value = error.message || 'Failed to parse file';
  } finally {
    isProcessing.value = false;
  }
};

const parseJSON = async (file: File) => {
  const text = await file.text();
  const data = JSON.parse(text);

  // Expect array of components or { components: [...] }
  const components = Array.isArray(data) ? data : data.components;

  if (!components || !Array.isArray(components)) {
    throw new Error('Invalid JSON structure. Expected array of components.');
  }

  parsedData.value = components;
  emit('update:modelValue', components);
  emit('validation-change', components.length >= 2);
};

const parseCSV = async (file: File) => {
  const text = await file.text();
  const lines = text.split('\n').filter(line => line.trim());

  if (lines.length < 2) {
    throw new Error('CSV file must have at least a header row and one data row.');
  }

  // Parse header
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  
  // Parse rows
  const components: Component[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    const component: any = {};

    headers.forEach((header, index) => {
      const value = values[index];
      
      // Map CSV columns to component properties
      if (header === 'componentid' || header === 'component_id') {
        component.componentId = value;
      } else if (header === 'componenttype' || header === 'component_type' || header === 'type') {
        component.componentType = value;
      } else if (header === 'sensors') {
        component.sensors = value.split(';').map(s => s.trim());
      } else if (header === 'nominalpressurebar' || header === 'pressure') {
        component.nominalPressureBar = parseFloat(value);
      } else if (header === 'nominalflowlpm' || header === 'flow') {
        component.nominalFlowLpm = parseFloat(value);
      } else if (header === 'ratedpowerkw' || header === 'power') {
        component.ratedPowerKw = parseFloat(value);
      }
    });

    if (component.componentId && component.componentType) {
      if (!component.sensors) component.sensors = ['pressure_in'];
      components.push(component as Component);
    }
  }

  if (components.length === 0) {
    throw new Error('No valid components found in CSV file.');
  }

  parsedData.value = components;
  emit('update:modelValue', components);
  emit('validation-change', components.length >= 2);
};

const clearFile = () => {
  uploadedFile.value = null;
  parsedData.value = [];
  parseError.value = null;
  if (fileInputRef.value) {
    fileInputRef.value.value = '';
  }
  emit('update:modelValue', []);
  emit('validation-change', false);
};
</script>

<template>
  <div class="space-y-6">
    <div>
      <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
        Import P&ID Schema
      </h3>
      <p class="text-sm text-gray-600 dark:text-gray-400">
        Upload a CSV or JSON file with component definitions to auto-populate the wizard.
      </p>
    </div>

    <!-- Upload Zone -->
    <Card
      v-if="!uploadedFile"
      class="border-2 border-dashed transition-colors"
      :class="[
        isDragging
          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
          : 'border-gray-300 dark:border-gray-600 hover:border-gray-400',
      ]"
      @dragover.prevent="handleDragOver"
      @dragleave="handleDragLeave"
      @drop.prevent="handleDrop"
    >
      <div class="text-center py-12">
        <Icon
          name="heroicons:cloud-arrow-up"
          class="mx-auto h-12 w-12 text-gray-400"
          :class="{ 'text-primary-500': isDragging }"
        />
        <div class="mt-4">
          <Button @click="openFilePicker">
            <Icon name="heroicons:document-arrow-up" class="h-5 w-5 mr-2" />
            Choose File
          </Button>
          <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
            or drag and drop
          </p>
        </div>
        <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
          CSV, JSON up to 10MB
        </p>
        <input
          ref="fileInputRef"
          type="file"
          :accept="acceptedFormats"
          class="hidden"
          @change="handleFileSelect"
        />
      </div>
    </Card>

    <!-- File Info & Preview -->
    <template v-else>
      <Card>
        <div class="flex items-start justify-between">
          <div class="flex gap-3">
            <Icon name="heroicons:document-text" class="h-8 w-8 text-primary-600 shrink-0" />
            <div>
              <h4 class="text-sm font-semibold text-gray-900 dark:text-gray-100">
                {{ fileInfo?.name }}
              </h4>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {{ fileInfo?.size }} • {{ fileInfo?.type || 'Unknown type' }}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" @click="clearFile">
            <Icon name="heroicons:x-mark" class="h-5 w-5" />
          </Button>
        </div>

        <!-- Processing State -->
        <div v-if="isProcessing" class="mt-4 flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <Icon name="heroicons:arrow-path" class="h-5 w-5 animate-spin" />
          <span>Processing file...</span>
        </div>

        <!-- Parse Error -->
        <div
          v-else-if="parseError"
          class="mt-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3"
        >
          <div class="flex gap-2">
            <Icon name="heroicons:x-circle" class="h-5 w-5 text-red-600 dark:text-red-400 shrink-0" />
            <p class="text-sm text-red-800 dark:text-red-200">
              {{ parseError }}
            </p>
          </div>
        </div>

        <!-- Success State -->
        <div
          v-else-if="parsedData.length > 0"
          class="mt-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 p-3"
        >
          <div class="flex gap-2">
            <Icon name="heroicons:check-circle" class="h-5 w-5 text-green-600 dark:text-green-400 shrink-0" />
            <div class="flex-1">
              <p class="text-sm font-medium text-green-900 dark:text-green-100">
                Successfully parsed {{ parsedData.length }} component{{ parsedData.length > 1 ? 's' : '' }}
              </p>
              <p class="text-xs text-green-700 dark:text-green-300 mt-1">
                You can proceed to the next step or manually edit components.
              </p>
            </div>
          </div>
        </div>
      </Card>

      <!-- Preview Table -->
      <Card v-if="parsedData.length > 0">
        <template #header>
          <h4 class="text-base font-semibold text-gray-900 dark:text-gray-100">
            Preview ({{ parsedData.length }} components)
          </h4>
        </template>

        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th class="px-3 py-2 text-left text-xs font-semibold text-gray-700 dark:text-gray-300">
                  ID
                </th>
                <th class="px-3 py-2 text-left text-xs font-semibold text-gray-700 dark:text-gray-300">
                  Type
                </th>
                <th class="px-3 py-2 text-left text-xs font-semibold text-gray-700 dark:text-gray-300">
                  Sensors
                </th>
                <th class="px-3 py-2 text-left text-xs font-semibold text-gray-700 dark:text-gray-300">
                  Pressure
                </th>
                <th class="px-3 py-2 text-left text-xs font-semibold text-gray-700 dark:text-gray-300">
                  Flow
                </th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 dark:divide-gray-700">
              <tr v-for="(component, index) in parsedData.slice(0, 5)" :key="index">
                <td class="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">
                  {{ component.componentId }}
                </td>
                <td class="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">
                  {{ component.componentType }}
                </td>
                <td class="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">
                  {{ component.sensors.join(', ') }}
                </td>
                <td class="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">
                  {{ component.nominalPressureBar || '-' }}
                </td>
                <td class="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">
                  {{ component.nominalFlowLpm || '-' }}
                </td>
              </tr>
            </tbody>
          </table>
          <p v-if="parsedData.length > 5" class="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
            Showing first 5 of {{ parsedData.length }} components
          </p>
        </div>
      </Card>
    </template>

    <!-- Format Examples -->
    <Card variant="outlined">
      <template #header>
        <h4 class="text-sm font-semibold text-gray-900 dark:text-gray-100">
          File Format Examples
        </h4>
      </template>

      <div class="space-y-4">
        <div>
          <p class="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">CSV Format:</p>
          <pre class="text-xs bg-gray-50 dark:bg-gray-800 p-3 rounded overflow-x-auto"><code>componentId,componentType,sensors,nominalPressureBar,nominalFlowLpm
pump_main_1,piston_pump,pressure_in;pressure_out;temperature,280,120
valve_main,directional_valve,position,250,100</code></pre>
        </div>

        <div>
          <p class="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">JSON Format:</p>
          <pre class="text-xs bg-gray-50 dark:bg-gray-800 p-3 rounded overflow-x-auto"><code>[{
  "componentId": "pump_main_1",
  "componentType": "piston_pump",
  "sensors": ["pressure_in", "pressure_out"],
  "nominalPressureBar": 280
}]</code></pre>
        </div>
      </div>
    </Card>
  </div>
</template>
