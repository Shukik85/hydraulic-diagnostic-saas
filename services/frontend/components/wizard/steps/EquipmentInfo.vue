<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import type { EquipmentMetadata } from '~/types/gnn';
import { useTopologyValidation } from '~/composables/useTopologyValidation';

interface Props {
  modelValue: EquipmentMetadata;
}

const props = defineProps<Props>();
const emit = defineEmits<{
  'update:modelValue': [value: EquipmentMetadata];
  'validation-change': [isValid: boolean];
}>();

const { validateEquipment } = useTopologyValidation();

const equipment = ref<EquipmentMetadata>({ ...props.modelValue });
const errors = ref<Record<string, string>>({});

// Watch for external changes
watch(
  () => props.modelValue,
  (newValue) => {
    equipment.value = { ...newValue };
  },
  { deep: true }
);

// Watch for internal changes and emit
watch(
  equipment,
  (newValue) => {
    emit('update:modelValue', newValue);
    validate();
  },
  { deep: true }
);

const validate = () => {
  const validationErrors = validateEquipment(equipment.value);
  
  // Convert array to object for easy access
  errors.value = {};
  validationErrors.forEach((err) => {
    errors.value[err.field] = err.message;
  });

  const isValid = validationErrors.length === 0;
  emit('validation-change', isValid);
  
  return isValid;
};

const isValid = computed(() => {
  return Object.keys(errors.value).length === 0;
});

// Validate on mount
validate();
</script>

<template>
  <div class="space-y-6">
    <div>
      <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
        Equipment Information
      </h3>
      <p class="text-sm text-gray-600 dark:text-gray-400">
        Enter basic information about the hydraulic equipment you want to onboard.
      </p>
    </div>

    <Card variant="outlined" padding="lg">
      <div class="space-y-4">
        <!-- Equipment ID -->
        <Input
          v-model="equipment.equipmentId"
          label="Equipment ID"
          placeholder="e.g., EXC-001"
          required
          :error="errors.equipmentId"
        >
          <template #description>
            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Unique identifier (1-100 characters, alphanumeric)
            </p>
          </template>
        </Input>

        <!-- Equipment Name -->
        <Input
          v-model="equipment.equipmentName"
          label="Equipment Name"
          placeholder="e.g., Komatsu PC200-8 Excavator"
          required
          :error="errors.equipmentName"
        >
          <template #description>
            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Full name or model (1-255 characters)
            </p>
          </template>
        </Input>

        <!-- Equipment Type -->
        <Input
          v-model="equipment.equipmentType"
          label="Equipment Type"
          placeholder="e.g., excavator, loader, crane"
          :error="errors.equipmentType"
        >
          <template #description>
            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Category or type of equipment
            </p>
          </template>
        </Input>

        <!-- Operating Hours -->
        <Input
          v-model.number="equipment.operatingHours"
          type="number"
          label="Operating Hours"
          placeholder="e.g., 5000"
          :error="errors.operatingHours"
        >
          <template #description>
            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Total hours of operation (optional)
            </p>
          </template>
        </Input>
      </div>
    </Card>

    <!-- Validation Summary -->
    <div
      v-if="!isValid"
      class="rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 p-4"
    >
      <div class="flex gap-3">
        <Icon name="heroicons:exclamation-triangle" class="h-5 w-5 text-amber-600 dark:text-amber-400 shrink-0" />
        <div>
          <h4 class="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-1">
            Please fix the following errors:
          </h4>
          <ul class="text-sm text-amber-800 dark:text-amber-200 list-disc list-inside space-y-1">
            <li v-for="(error, field) in errors" :key="field">
              {{ error }}
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>
