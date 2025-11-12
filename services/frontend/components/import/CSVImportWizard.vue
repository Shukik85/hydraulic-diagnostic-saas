<template>
  <div class="csv-import-wizard">
    <h2 class="text-xl font-semibold mb-4">Импорт данных с датчиков (CSV Wizard)</h2>
    <UStepper v-model="step" :items="steps" class="mb-8" />
    <template v-if="step === 0">
      <!-- Step 1: Upload CSV -->
      <BaseCard padding="lg">
        <div class="flex flex-col items-center gap-6 py-8">
          <input
            ref="fileInput"
            type="file"
            accept=".csv"
            class="hidden"
            @change="onFileChange"
          />
          <BaseButton
            icon="heroicons:arrow-up-tray"
            size="lg"
            @click="$refs.fileInput.click()"
            >Загрузить CSV файл</BaseButton
          >
          <span v-if="selectedFileName" class="text-industrial-500">{{ selectedFileName }}</span>
        </div>
      </BaseCard>
    </template>
    <template v-else-if="step === 1">
      <!-- Step 2: Validate CSV -->
      <BaseCard>
        <div v-if="validationState === 'loading'" class="p-6 flex justify-center">
          <Icon name="svg-spinners:blocks-wave" class="w-7 h-7 text-hydraulic-500 animate-spin" />
          <span class="ml-3">Валидация файла...</span>
        </div>
        <div v-else-if="validation.state === 'error'" class="p-4 bg-red-100 rounded text-red-700">
          <div v-for="err in validationResult.errors" :key="err.field">❌ {{ err.message }}</div>
        </div>
        <div v-else>
          <div v-if="validationResult.errors.length">
            <UAlert color="red" v-for="err in validationResult.errors" :key="err.field">
              {{ err.message }}
            </UAlert>
          </div>
          <div v-if="validationResult.warnings.length">
            <UAlert color="yellow" v-for="w in validationResult.warnings" :key="w.sensor">
              ⚠️ {{ w.message }}
            </UAlert>
          </div>
          <BaseButton
            v-if="validationResult.valid"
            :disabled="validationResult.errors.length > 0"
            @click="step++"
            variant="success"
          >
            Продолжить
          </BaseButton>
        </div>
      </BaseCard>
    </template>
    <template v-else-if="step === 2">
      <!-- Step 3: Preview -->
      <BaseCard>
        <h3 class="font-medium mb-2">Просмотр данных (первые 10 строк)</h3>
        <UTable :rows="validationResult.preview" />
        <div class="text-sm text-industrial-500 mt-3">Всего строк: {{ validationResult.stats.rows_total }}</div>
        <BaseButton class="mt-6" @click="step++">Импортировать</BaseButton>
      </BaseCard>
    </template>
    <template v-else-if="step === 3">
      <!-- Step 4: Import -->
      <BaseCard class="flex flex-col gap-6 items-center py-10">
        <div v-if="importState === 'progress'">
          <UProgress :value="importProgress" class="w-full max-w-lg" />
          <div class="text-industrial-500 mt-4">Импорт данных...</div>
        </div>
        <div v-else-if="importState === 'done'">
          <UAlert color="green">Данные успешно импортированы!</UAlert>
          <BaseButton class="mt-4" variant="success" @click="finish">Завершить</BaseButton>
        </div>
        <div v-else-if="importState === 'error'">
          <UAlert color="red">Произошла ошибка импорта</UAlert>
        </div>
      </BaseCard>
    </template>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'
import { useToast } from '~/composables/useToast'
import { useErrorHandler } from '~/composables/useErrorHandler'

const store = useMetadataStore()
const toast = useToast()
const errorHandler = useErrorHandler()
const { post } = useApi()

const step = ref(0)
const steps = [
  'Загрузка CSV',
  'Валидация',
  'Просмотр',
  'Импорт',
]

const fileInput = ref<HTMLInputElement>()
const selectedFile = ref<File | null>(null)
const selectedFileName = ref('')

const validationState = ref<'idle' | 'loading' | 'done' | 'error'>('idle')
const validationResult = ref({
  valid: false,
  errors: [],
  warnings: [],
  preview: [],
  stats: {} as any
})

async function onFileChange(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  selectedFile.value = file
  selectedFileName.value = file.name
  step.value = 1
  await validateCSV()
}

async function validateCSV() {
  validationState.value = 'loading'
  const equipmentId = store.wizardState.system.equipment_id
  const formData = new FormData()
  formData.append('file', selectedFile.value!)
  formData.append('equipment_id', equipmentId)

  try {
    const response = await post('/api/csv-upload/validate', formData)
    if (isApiSuccess(response)) {
      validationResult.value = response.data
      validationState.value = 'done'
    } else {
      validationResult.value = {
        valid: false,
        errors: response.error.errors || [response.error],
        warnings: [],
        preview: [],
        stats: {}
      }
      validationState.value = 'error'
    }
  } catch (error) {
    errorHandler.handleApiError(error, 'Валидация CSV')
    validationState.value = 'error'
  }
}

const importState = ref<'idle' | 'progress' | 'done' | 'error'>('idle')
const importProgress = ref(0)

async function doImport() {
  importState.value = 'progress'
  const equipmentId = store.wizardState.system.equipment_id
  const formData = new FormData()
  formData.append('file', selectedFile.value!)
  formData.append('equipment_id', equipmentId)

  try {
    const response = await post('/api/csv-upload/import', formData)
    if (isApiSuccess(response)) {
      importProgress.value = 100
      importState.value = 'done'
      toast.success(`Импортировано ${response.data.imported_readings} записей`)
    } else {
      importState.value = 'error'
      errorHandler.handleApiError(response, 'Импорт CSV')
    }
  } catch (error) {
    importState.value = 'error'
    errorHandler.handleApiError(error, 'Импорт CSV')
  }
}

function finish() {
  step.value = 0
  selectedFile.value = null
  selectedFileName.value = ''
  validationResult.value = {
    valid: false,
    errors: [],
    warnings: [],
    preview: [],
    stats: {}
  }
}
</script>

<style scoped>
.csv-import-wizard {
  @apply p-4;
}
</style>