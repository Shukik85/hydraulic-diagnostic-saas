<!-- components/metadata/Level5Validation.vue -->
<template>
  <div class="level-5 space-y-6">
    <div>
      <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
        {{ $t('wizard.level5.title') }}
      </h2>
      <p class="text-sm text-gray-600 dark:text-gray-400">
        {{ $t('wizard.level5.description') }}
      </p>
    </div>

    <!-- Overall Progress -->
    <UCard class="p-8">
      <div class="flex flex-col items-center">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
          {{ $t('wizard.level5.overallReadiness') }}
        </h3>
        
        <!-- Progress circle -->
        <div class="relative w-32 h-32 mb-6">
          <svg class="w-32 h-32 transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="54"
              fill="none"
              stroke="currentColor"
              stroke-width="8"
              class="text-gray-200 dark:text-gray-700"
            />
            <circle
              cx="64"
              cy="64"
              r="54"
              fill="none"
              :stroke="progressColor"
              stroke-width="8"
              stroke-linecap="round"
              :stroke-dasharray="circumference"
              :stroke-dashoffset="dashOffset"
              class="transition-all duration-500"
            />
          </svg>
          <div class="absolute inset-0 flex flex-col items-center justify-center">
            <span class="text-3xl font-bold text-gray-900 dark:text-gray-100">
              {{ store.completeness }}%
            </span>
            <span class="text-xs text-gray-500 dark:text-gray-400">
              {{ $t('wizard.level5.ready') }}
            </span>
          </div>
        </div>
        
        <!-- Status message -->
        <UAlert
          v-if="store.completeness < 50"
          color="red"
          icon="i-heroicons-x-circle"
          :title="$t('wizard.level5.insufficient')"
          :description="$t('wizard.level5.insufficientDesc')"
          class="max-w-md"
        />
        <UAlert
          v-else-if="store.completeness < 70"
          color="yellow"
          icon="i-heroicons-exclamation-triangle"
          :title="$t('wizard.level5.good')"
          :description="$t('wizard.level5.goodDesc')"
          class="max-w-md"
        />
        <UAlert
          v-else
          color="green"
          icon="i-heroicons-check-circle"
          :title="$t('wizard.level5.excellent')"
          :description="$t('wizard.level5.excellentDesc')"
          class="max-w-md"
        />
      </div>
    </UCard>

    <!-- Validation Errors -->
    <UCard v-if="validationErrors.length > 0" class="p-6">
      <div class="flex items-center gap-2 mb-4">
        <UIcon name="i-heroicons-x-circle" class="w-5 h-5 text-red-600 dark:text-red-400" />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ $t('wizard.level5.validationErrors') }} ({{ validationErrors.length }})
        </h3>
      </div>
      
      <div class="space-y-3">
        <div
          v-for="(error, i) in validationErrors"
          :key="i"
          class="flex gap-3 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg"
        >
          <span class="text-xl">⚠</span>
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-900 dark:text-gray-100">
              {{ error.error }}
            </p>
            <p v-if="error.suggestion" class="text-xs text-gray-600 dark:text-gray-400 mt-1">
              {{ error.suggestion }}
            </p>
          </div>
        </div>
      </div>
    </UCard>

    <!-- Missing Critical Fields -->
    <UCard v-if="incompleteness.critical_missing.length > 0" class="p-6">
      <div class="flex items-center gap-2 mb-4">
        <UIcon name="i-heroicons-exclamation-triangle" class="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ $t('wizard.level5.criticalFields') }} ({{ incompleteness.critical_missing.length }})
        </h3>
      </div>
      <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {{ $t('wizard.level5.criticalFieldsDesc') }}
      </p>
      
      <div class="space-y-2">
        <div
          v-for="field in incompleteness.critical_missing"
          :key="field"
          class="flex items-center gap-2 text-sm"
        >
          <div class="w-2 h-2 rounded-full bg-red-500" />
          <span class="text-gray-700 dark:text-gray-300">{{ field }}</span>
        </div>
      </div>
    </UCard>

    <!-- Missing Secondary Fields -->
    <UCard v-if="incompleteness.secondary_missing.length > 0" class="p-6">
      <div class="flex items-center gap-2 mb-4">
        <UIcon name="i-heroicons-information-circle" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ $t('wizard.level5.secondaryFields') }} ({{ incompleteness.secondary_missing.length }})
        </h3>
      </div>
      <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {{ $t('wizard.level5.secondaryFieldsDesc') }}
      </p>
      
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <div
          v-for="field in incompleteness.secondary_missing"
          :key="field"
          class="flex items-center gap-2 text-sm"
        >
          <div class="w-2 h-2 rounded-full bg-yellow-500" />
          <span class="text-gray-700 dark:text-gray-300">{{ field }}</span>
        </div>
      </div>
    </UCard>

    <!-- Inferred Values -->
    <UCard v-if="Object.keys(incompleteness.inferred_values).length > 0" class="p-6">
      <div class="flex items-center gap-2 mb-4">
        <UIcon name="i-heroicons-sparkles" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ $t('wizard.level5.inferredValues') }} ({{ Object.keys(incompleteness.inferred_values).length }})
        </h3>
      </div>
      <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {{ $t('wizard.level5.inferredValuesDesc') }}
      </p>
      
      <div class="space-y-3">
        <div
          v-for="(data, field) in incompleteness.inferred_values"
          :key="field"
          class="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg"
        >
          <div class="flex items-start gap-2">
            <span class="text-xl">💡</span>
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-gray-100">
                {{ field }}: {{ JSON.stringify(data.value) }}
              </p>
              <div class="flex items-center gap-3 mt-2 text-xs">
                <span class="text-gray-600 dark:text-gray-400 italic">
                  {{ data.method }}
                </span>
                <UBadge
                  :color="getConfidenceColor(data.confidence)"
                  variant="soft"
                  size="xs"
                >
                  {{ $t('wizard.level5.confidence') }}: {{ (data.confidence * 100).toFixed(0) }}%
                </UBadge>
              </div>
            </div>
          </div>
        </div>
      </div>
    </UCard>

    <!-- Summary Stats -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
      <UCard class="p-6">
        <div class="text-center">
          <p class="text-3xl font-bold text-blue-600 dark:text-blue-400">
            {{ store.componentsCount }}
          </p>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-2">
            {{ $t('wizard.level5.summary.components') }}
          </p>
        </div>
      </UCard>
      
      <UCard class="p-6">
        <div class="text-center">
          <p class="text-3xl font-bold text-green-600 dark:text-green-400">
            {{ adjacencyEdgesCount }}
          </p>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-2">
            {{ $t('wizard.level5.summary.connections') }}
          </p>
        </div>
      </UCard>
      
      <UCard class="p-6">
        <div class="text-center">
          <p class="text-3xl font-bold text-purple-600 dark:text-purple-400">
            {{ store.completeness }}%
          </p>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-2">
            {{ $t('wizard.level5.summary.filled') }}
          </p>
        </div>
      </UCard>
      
      <UCard class="p-6">
        <div class="text-center">
          <p class="text-3xl font-bold text-indigo-600 dark:text-indigo-400">
            {{ confidenceScore.toFixed(2) }}
          </p>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-2">
            {{ $t('wizard.level5.summary.confidence') }}
          </p>
        </div>
      </UCard>
    </div>

    <!-- Actions -->
    <div class="flex justify-end gap-3">
      <UButton
        color="gray"
        variant="outline"
        icon="i-heroicons-sparkles"
        :disabled="isSubmitting"
        @click="runInference"
      >
        {{ $t('wizard.level5.actions.inferValues') }}
      </UButton>

      <UButton
        color="primary"
        icon="i-heroicons-check"
        :loading="isSubmitting"
        :disabled="!canSubmit"
        size="lg"
        @click="submitMetadata"
      >
        {{ store.completeness >= 70 ? $t('wizard.level5.actions.submit') : $t('wizard.level5.actions.submitWithGaps') }}
      </UButton>
    </div>

    <!-- Result Modal -->
    <UModal v-model="showResultModal">
      <UCard>
        <div v-if="submitSuccess" class="text-center p-6">
          <div class="text-6xl mb-4">✅</div>
          <h3 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            {{ $t('wizard.level5.modal.successTitle') }}
          </h3>
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-6">
            {{ $t('wizard.level5.modal.successDesc') }}<br>
            {{ $t('wizard.level5.modal.dataCompleteness') }}: {{ store.completeness }}%
          </p>
          <UButton color="primary" @click="goToDashboard">
            {{ $t('wizard.level5.modal.goToDashboard') }}
          </UButton>
        </div>

        <div v-else class="text-center p-6">
          <div class="text-6xl mb-4">❌</div>
          <h3 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            {{ $t('wizard.level5.modal.errorTitle') }}
          </h3>
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-6">
            {{ submitError }}
          </p>
          <UButton color="gray" @click="showResultModal = false">
            {{ $t('ui.close') }}
          </UButton>
        </div>
      </UCard>
    </UModal>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'

const { t } = useI18n()
const store = useMetadataStore()
const router = useRouter()
const toast = useToast()

const isSubmitting = ref(false)
const showResultModal = ref(false)
const submitSuccess = ref(false)
const submitError = ref('')

const validationErrors = computed(() => store.validateConsistency())
const incompleteness = computed(() => store.wizardState.incompleteness_report)

const adjacencyEdgesCount = computed(() => {
  const matrix = store.wizardState.system.adjacency_matrix || []
  return matrix.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0)
})

const confidenceScore = computed(() => {
  const inferred = Object.values(incompleteness.value.inferred_values)
  if (inferred.length === 0) return 1.0
  const avgConfidence = inferred.reduce((sum, v) => sum + (v as any).confidence, 0) / inferred.length
  return avgConfidence
})

const canSubmit = computed(() => {
  return validationErrors.value.length === 0 && store.componentsCount > 0
})

const circumference = 2 * Math.PI * 54
const dashOffset = computed(() => {
  return circumference - (store.completeness / 100) * circumference
})

const progressColor = computed(() => {
  if (store.completeness < 50) return '#ef4444'
  if (store.completeness < 70) return '#f59e0b'
  return '#10b981'
})

function getConfidenceColor(confidence: number): string {
  if (confidence < 0.5) return 'red'
  if (confidence < 0.7) return 'yellow'
  return 'green'
}

function runInference() {
  store.inferMissingValues()
  toast.add({
    title: t('wizard.level5.actions.inferValues'),
    description: t('wizard.level5.inferredValuesDesc'),
    color: 'green'
  })
}

async function submitMetadata() {
  isSubmitting.value = true

  try {
    const result = await store.submitMetadata()

    if (result.success) {
      submitSuccess.value = true
      showResultModal.value = true
    } else {
      submitSuccess.value = false
      const error = result.error as any
      submitError.value = error?.message || t('wizard.level5.modal.errorTitle')
      showResultModal.value = true
    }
  } catch (error: any) {
    submitSuccess.value = false
    submitError.value = error.message || t('wizard.level5.modal.errorTitle')
    showResultModal.value = true
  } finally {
    isSubmitting.value = false
  }
}

function goToDashboard() {
  router.push('/dashboard')
}
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
