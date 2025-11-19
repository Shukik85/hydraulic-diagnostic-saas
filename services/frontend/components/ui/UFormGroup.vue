<template>
  <div :class="['form-group', groupClass]">
    <!-- Label -->
    <ULabel 
      v-if="label"
      :for="inputId"
      class="form-label mb-2"
    >
      {{ label }}
      <span v-if="required" class="text-red-400 ml-1">*</span>
    </ULabel>

    <!-- Input Slot -->
    <div class="relative">
      <slot />

      <!-- Error Icon -->
      <div 
        v-if="error"
        class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
      >
        <Icon 
          name="heroicons:exclamation-circle" 
          class="w-5 h-5 text-red-400"
        />
      </div>
    </div>

    <!-- Helper Text -->
    <UHelperText 
      v-if="helper && !error"
      :text="helper"
      variant="default"
      class="mt-1.5"
    />

    <!-- Error Message -->
    <UHelperText 
      v-if="error"
      :text="error"
      variant="error"
      class="mt-1.5"
    />
  </div>
</template>

<script setup lang="ts">
interface Props {
  label?: string
  helper?: string
  error?: string
  required?: boolean
  inputId?: string
  groupClass?: string
}

withDefaults(defineProps<Props>(), {
  label: undefined,
  helper: undefined,
  error: undefined,
  required: false,
  inputId: undefined,
  groupClass: '',
})
</script>

<style scoped>
.form-group {
  @apply space-y-2;
}

.form-label {
  @apply block text-sm font-medium text-white;
}
</style>
