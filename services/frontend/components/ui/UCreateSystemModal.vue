<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'

const loading = ref(false)
const modelValue = ref(true)
const { t } = useI18n()

const form = ref({
  name: '',
  type: 'industrial',
  status: 'active',
  description: ''
})

const errors = ref<{ name?: string }>({})

function validate() {
  errors.value.name = !form.value.name ? t('systems.create.nameRequired') : ''
  return !errors.value.name
}

const isValid = computed(validate)

function handleCancel() {
  modelValue.value = false
}

function handleSubmit() {
  if (!isValid.value) return
  loading.value = true
  setTimeout(() => loading.value = false, 1500)
}

</script>
<template>
  <UModal
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :title="t('systems.create.title')"
    :description="t('systems.create.subtitle')"
    size="md"
    :close-on-backdrop="true"
  >
    <div class="space-y-5">
      <!-- System Name -->
      <div>
        <label class="u-label" for="system-name">{{ t('systems.create.name') }} *</label>
        <input id="system-name" v-model.trim="form.name" type="text" class="u-input" :placeholder="t('systems.create.namePlaceholder')" :disabled="loading" maxlength="200" />
        <Transition name="fade">
          <p v-if="errors.name" class="mt-2 text-sm text-error-500 flex items-center gap-1">
            <Icon name="heroicons:exclamation-circle" class="h-4 w-4 shrink-0" />
            {{ errors.name }}
          </p>
        </Transition>
      </div>
      <!-- System Type -->
      <div class="relative">
        <label class="u-label" for="system-type">{{ t('systems.create.type') }}</label>
        <select id="system-type" v-model="form.type" class="u-input metallic-select" :disabled="loading">
          <option value="industrial">{{ t('systems.types.industrial') }}</option>
          <option value="mobile">{{ t('systems.types.mobile') }}</option>
          <option value="marine">{{ t('systems.types.marine') }}</option>
          <option value="construction">{{ t('systems.types.construction') }}</option>
          <option value="mining">{{ t('systems.types.mining') }}</option>
          <option value="agricultural">{{ t('systems.types.agricultural') }}</option>
        </select>
        <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-steel-light pointer-events-none" />
      </div>
      <!-- Initial Status -->
      <div class="relative">
        <label class="u-label" for="system-status">{{ t('systems.create.initialStatus') }}</label>
        <select id="system-status" v-model="form.status" class="u-input metallic-select" :disabled="loading">
          <option value="active">{{ t('systems.status.active') }}</option>
          <option value="maintenance">{{ t('systems.status.maintenance') }}</option>
          <option value="inactive">{{ t('systems.status.inactive') }}</option>
        </select>
        <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-steel-light pointer-events-none" />
      </div>
      <!-- Description -->
      <div>
        <label class="u-label" for="system-description">{{ t('ui.description') }} <span class="text-text-secondary font-normal">({{ t('ui.optional') }})</span></label>
        <textarea id="system-description" v-model.trim="form.description" class="u-input resize-none" :placeholder="t('systems.create.descriptionPlaceholder')" :disabled="loading" rows="3" maxlength="500" />
      </div>
      <!-- Setup Info -->
      <div class="rounded-lg bg-primary-500/5 border border-primary-500/30 p-4">
        <div class="flex items-start gap-3">
          <Icon name="heroicons:information-circle" class="h-5 w-5 text-primary-400 mt-0.5 shrink-0" />
          <div>
            <p class="text-sm font-medium text-primary-900">{{ t('systems.create.nextStepsTitle') }}</p>
            <p class="text-sm text-primary-600 mt-1">{{ t('systems.create.nextStepsDesc') }}</p>
          </div>
        </div>
      </div>
    </div>
    <template #footer>
      <button class="u-btn u-btn-secondary" @click="handleCancel" :disabled="loading" type="button">{{ t('ui.cancel') }}</button>
      <button class="u-btn u-btn-primary min-w-[120px]" @click="handleSubmit" :disabled="!isValid || loading" type="button">
        <Icon v-if="loading" name="heroicons:arrow-path" class="h-4 w-4 animate-spin mr-2" />
        <Icon v-else name="heroicons:plus" class="h-4 w-4 mr-2" />
        {{ loading ? t('systems.create.creating') : t('systems.create.createBtn') }}
      </button>
    </template>
  </UModal>
</template>
<style scoped>.fade-enter-active, .fade-leave-active { transition: all 0.15s ease-out; } .fade-enter-from, .fade-leave-to { opacity: 0; transform: translateY(-4px); } .metallic-select { background-color: #191d23 !important; color: #edf2fa !important; border-color: #4c596f !important; }</style>
