<script setup lang="ts">
import { ref, computed } from 'vue';

const { t } = useI18n();
const toast = useToast();

const currentStep = ref(1);

interface WizardStep {
  level: number;
  title: string;
  description: string;
  completed: boolean;
}

const steps = ref<WizardStep[]>([
  {
    level: 1,
    title: t('wizard.level1.title'),
    description: t('wizard.level1.description'),
    completed: false,
  },
  {
    level: 2,
    title: t('wizard.level2.title'),
    description: t('wizard.level2.description'),
    completed: false,
  },
  {
    level: 3,
    title: t('wizard.level3.title'),
    description: t('wizard.level3.description'),
    completed: false,
  },
  {
    level: 4,
    title: t('wizard.level4.title'),
    description: t('wizard.level4.description'),
    completed: false,
  },
  {
    level: 5,
    title: t('wizard.level5.title'),
    description: t('wizard.level5.description'),
    completed: false,
  },
]);

// Level 1: P&ID Schema
const schemaFile = ref<File | null>(null);
const schemaFormat = ref<'pdf' | 'svg' | 'png' | 'jpg'>('pdf');

// Level 2: Sensors
const sensors = ref<Array<{ id: string; type: string; name: string }>>([
  { id: '1', type: 'pressure', name: 'Pressure Sensor 1' },
]);

// Level 3: Nominal Values
const nominalValues = ref<Record<string, { min: number; max: number; nominal: number }>>({
  '1': { min: 0, max: 200, nominal: 150 },
});

// Level 4: Operating Modes
const operatingModes = ref<string[]>(['Normal', 'Startup', 'Shutdown']);

// Level 5: AI Readiness
const aiReadiness = ref(0);

const currentStepData = computed(() => steps.value[currentStep.value - 1]);

const canProceed = computed(() => {
  switch (currentStep.value) {
    case 1:
      return schemaFile.value !== null;
    case 2:
      return sensors.value.length > 0;
    case 3:
      return Object.keys(nominalValues.value).length > 0;
    case 4:
      return operatingModes.value.length > 0;
    case 5:
      return true;
    default:
      return false;
  }
});

const handleFileUpload = (event: Event): void => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files[0]) {
    schemaFile.value = target.files[0];
  }
};

const addSensor = (): void => {
  const id = `${sensors.value.length + 1}`;
  sensors.value.push({
    id,
    type: 'pressure',
    name: `Sensor ${id}`,
  });
  nominalValues.value[id] = { min: 0, max: 100, nominal: 50 };
};

const removeSensor = (id: string): void => {
  sensors.value = sensors.value.filter((s) => s.id !== id);
  delete nominalValues.value[id];
};

const addOperatingMode = (): void => {
  const name = prompt(t('wizard.level4.addModePrompt'));
  if (name) {
    operatingModes.value.push(name);
  }
};

const calculateAIReadiness = (): void => {
  let score = 0;
  if (schemaFile.value) score += 20;
  score += Math.min(sensors.value.length * 10, 30);
  if (Object.keys(nominalValues.value).length > 0) score += 25;
  score += Math.min(operatingModes.value.length * 5, 25);
  aiReadiness.value = score;
};

const nextStep = (): void => {
  if (!canProceed.value) {
    toast.warning(t('wizard.completeStep'), '');
    return;
  }

  steps.value[currentStep.value - 1].completed = true;

  if (currentStep.value < 5) {
    currentStep.value++;
  } else {
    calculateAIReadiness();
    toast.success(t('wizard.completed'), '');
  }
};

const previousStep = (): void => {
  if (currentStep.value > 1) {
    currentStep.value--;
  }
};

const goToStep = (level: number): void => {
  if (level <= currentStep.value || steps.value[level - 2]?.completed) {
    currentStep.value = level;
  }
};
</script>

<template>
  <div class="h-full rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <!-- Progress Indicator -->
    <div class="border-b border-gray-200 p-6 dark:border-gray-700">
      <ProgressIndicator :steps="steps" :current-step="currentStep" @go-to-step="goToStep" />
    </div>

    <!-- Step Content -->
    <div class="min-h-[500px] p-6">
      <!-- Level 1: P&ID Schema -->
      <WizardStep
        v-if="currentStep === 1"
        :level="1"
        :title="currentStepData.title"
        :description="currentStepData.description"
      >
        <div class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {{ $t('wizard.level1.selectFormat') }}
            </label>
            <select
              v-model="schemaFormat"
              class="mt-2 w-full rounded-lg border border-gray-300 bg-white px-4 py-2 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            >
              <option value="pdf">PDF</option>
              <option value="svg">SVG</option>
              <option value="png">PNG</option>
              <option value="jpg">JPG</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {{ $t('wizard.level1.uploadSchema') }}
            </label>
            <div class="mt-2">
              <input
                type="file"
                :accept="`.${schemaFormat}`"
                @change="handleFileUpload"
                class="block w-full text-sm text-gray-500 file:mr-4 file:rounded-lg file:border-0 file:bg-primary-50 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-primary-700 hover:file:bg-primary-100 dark:text-gray-400 dark:file:bg-primary-900/20 dark:file:text-primary-400"
              />
            </div>
            <p v-if="schemaFile" class="mt-2 text-sm text-green-600 dark:text-green-400">
              ✓ {{ schemaFile.name }}
            </p>
          </div>
        </div>
      </WizardStep>

      <!-- Level 2: Sensor Placement -->
      <WizardStep
        v-else-if="currentStep === 2"
        :level="2"
        :title="currentStepData.title"
        :description="currentStepData.description"
      >
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h4 class="text-sm font-medium text-gray-700 dark:text-gray-300">
              {{ $t('wizard.level2.sensors') }} ({{ sensors.length }})
            </h4>
            <Button variant="primary" size="sm" @click="addSensor">
              <Icon name="heroicons:plus" class="h-4 w-4" />
              {{ $t('wizard.level2.addSensor') }}
            </Button>
          </div>

          <div class="space-y-3">
            <div
              v-for="sensor in sensors"
              :key="sensor.id"
              class="flex items-center gap-4 rounded-lg border border-gray-200 p-4 dark:border-gray-600"
            >
              <Icon name="heroicons:cpu-chip" class="h-5 w-5 text-primary-600 dark:text-primary-400" />
              <div class="flex-1">
                <Input v-model="sensor.name" :placeholder="$t('wizard.level2.sensorName')" />
              </div>
              <select
                v-model="sensor.type"
                class="rounded-lg border border-gray-300 px-3 py-2 dark:border-gray-600 dark:bg-gray-700"
              >
                <option value="pressure">{{ $t('sensors.pressure') }}</option>
                <option value="temperature">{{ $t('sensors.temperature') }}</option>
                <option value="flow">{{ $t('sensors.flow') }}</option>
                <option value="vibration">{{ $t('sensors.vibration') }}</option>
              </select>
              <button
                @click="removeSensor(sensor.id)"
                class="rounded-lg p-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                :aria-label="$t('common.delete')"
              >
                <Icon name="heroicons:trash" class="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </WizardStep>

      <!-- Level 3: Nominal Values -->
      <WizardStep
        v-else-if="currentStep === 3"
        :level="3"
        :title="currentStepData.title"
        :description="currentStepData.description"
      >
        <div class="space-y-4">
          <div
            v-for="sensor in sensors"
            :key="sensor.id"
            class="rounded-lg border border-gray-200 p-4 dark:border-gray-600"
          >
            <h5 class="mb-3 font-medium text-gray-900 dark:text-white">{{ sensor.name }}</h5>
            <div class="grid gap-4 sm:grid-cols-3">
              <div>
                <label class="block text-xs text-gray-600 dark:text-gray-400">{{ $t('wizard.level3.min') }}</label>
                <input
                  v-model.number="nominalValues[sensor.id].min"
                  type="number"
                  class="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 dark:border-gray-600 dark:bg-gray-700"
                />
              </div>
              <div>
                <label class="block text-xs text-gray-600 dark:text-gray-400">{{ $t('wizard.level3.nominal') }}</label>
                <input
                  v-model.number="nominalValues[sensor.id].nominal"
                  type="number"
                  class="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 dark:border-gray-600 dark:bg-gray-700"
                />
              </div>
              <div>
                <label class="block text-xs text-gray-600 dark:text-gray-400">{{ $t('wizard.level3.max') }}</label>
                <input
                  v-model.number="nominalValues[sensor.id].max"
                  type="number"
                  class="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 dark:border-gray-600 dark:bg-gray-700"
                />
              </div>
            </div>
          </div>
        </div>
      </WizardStep>

      <!-- Level 4: Operating Modes -->
      <WizardStep
        v-else-if="currentStep === 4"
        :level="4"
        :title="currentStepData.title"
        :description="currentStepData.description"
      >
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h4 class="text-sm font-medium text-gray-700 dark:text-gray-300">
              {{ $t('wizard.level4.modes') }} ({{ operatingModes.length }})
            </h4>
            <Button variant="primary" size="sm" @click="addOperatingMode">
              <Icon name="heroicons:plus" class="h-4 w-4" />
              {{ $t('wizard.level4.addMode') }}
            </Button>
          </div>

          <div class="grid gap-3 sm:grid-cols-2">
            <div
              v-for="(mode, index) in operatingModes"
              :key="index"
              class="flex items-center justify-between rounded-lg border border-gray-200 p-4 dark:border-gray-600"
            >
              <div class="flex items-center gap-3">
                <Icon name="heroicons:cog-6-tooth" class="h-5 w-5 text-primary-600 dark:text-primary-400" />
                <span class="font-medium text-gray-900 dark:text-white">{{ mode }}</span>
              </div>
              <button
                @click="operatingModes.splice(index, 1)"
                class="rounded-lg p-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
              >
                <Icon name="heroicons:trash" class="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </WizardStep>

      <!-- Level 5: AI Readiness -->
      <WizardStep
        v-else-if="currentStep === 5"
        :level="5"
        :title="currentStepData.title"
        :description="currentStepData.description"
      >
        <div class="space-y-6">
          <div class="rounded-lg bg-gradient-to-br from-primary-50 to-blue-50 p-6 dark:from-primary-900/20 dark:to-blue-900/20">
            <div class="mb-4 flex items-center justify-center">
              <div class="relative h-32 w-32">
                <svg class="h-full w-full" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="8"
                    class="text-gray-200 dark:text-gray-700"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="8"
                    class="text-primary-600 transition-all duration-1000 dark:text-primary-400"
                    :stroke-dasharray="`${aiReadiness * 2.83} 283`"
                    stroke-linecap="round"
                    transform="rotate(-90 50 50)"
                  />
                </svg>
                <div class="absolute inset-0 flex items-center justify-center">
                  <span class="text-3xl font-bold text-primary-600 dark:text-primary-400">
                    {{ aiReadiness }}%
                  </span>
                </div>
              </div>
            </div>

            <h4 class="mb-4 text-center text-xl font-bold text-gray-900 dark:text-white">
              {{ aiReadiness >= 80 ? $t('wizard.level5.excellent') : aiReadiness >= 50 ? $t('wizard.level5.good') : $t('wizard.level5.needsWork') }}
            </h4>

            <div class="space-y-2">
              <div class="flex items-center justify-between text-sm">
                <span class="text-gray-600 dark:text-gray-400">{{ $t('wizard.level5.schemaUploaded') }}</span>
                <Icon
                  :name="schemaFile ? 'heroicons:check-circle' : 'heroicons:x-circle'"
                  :class="schemaFile ? 'text-green-600' : 'text-gray-400'"
                  class="h-5 w-5"
                />
              </div>
              <div class="flex items-center justify-between text-sm">
                <span class="text-gray-600 dark:text-gray-400">{{ $t('wizard.level5.sensorsConfigured') }}</span>
                <span class="font-medium text-gray-900 dark:text-white">{{ sensors.length }}</span>
              </div>
              <div class="flex items-center justify-between text-sm">
                <span class="text-gray-600 dark:text-gray-400">{{ $t('wizard.level5.nominalValuesDefined') }}</span>
                <Icon name="heroicons:check-circle" class="h-5 w-5 text-green-600" />
              </div>
              <div class="flex items-center justify-between text-sm">
                <span class="text-gray-600 dark:text-gray-400">{{ $t('wizard.level5.operatingModes') }}</span>
                <span class="font-medium text-gray-900 dark:text-white">{{ operatingModes.length }}</span>
              </div>
            </div>
          </div>
        </div>
      </WizardStep>
    </div>

    <!-- Navigation -->
    <div class="border-t border-gray-200 p-6 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <Button
          v-if="currentStep > 1"
          variant="secondary"
          @click="previousStep"
        >
          <Icon name="heroicons:arrow-left" class="h-5 w-5" />
          {{ $t('wizard.previous') }}
        </Button>
        <div v-else />

        <Button
          variant="primary"
          :disabled="!canProceed"
          @click="nextStep"
        >
          {{ currentStep < 5 ? $t('wizard.next') : $t('wizard.finish') }}
          <Icon v-if="currentStep < 5" name="heroicons:arrow-right" class="h-5 w-5" />
          <Icon v-else name="heroicons:check" class="h-5 w-5" />
        </Button>
      </div>
    </div>
  </div>
</template>