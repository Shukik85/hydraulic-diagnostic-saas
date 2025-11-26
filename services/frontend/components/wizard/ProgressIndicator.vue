<script setup lang="ts">
import { computed } from 'vue';

interface Step {
  level: number;
  title: string;
  description: string;
  completed: boolean;
}

interface Props {
  steps: Step[];
  currentStep: number;
}

const props = defineProps<Props>();

const progress = computed(() => {
  const completedSteps = props.steps.filter(s => s.completed).length;
  return (completedSteps / props.steps.length) * 100;
});

const getStepStatus = (step: Step): 'completed' | 'current' | 'upcoming' => {
  if (step.completed) return 'completed';
  if (step.level === props.currentStep) return 'current';
  return 'upcoming';
};
</script>

<template>
  <div class="w-full">
    <!-- Progress Bar -->
    <div class="mb-8">
      <div class="mb-2 flex items-center justify-between">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">
          {{ $t('wizard.progress') }}
        </span>
        <span class="text-sm font-medium text-primary-600 dark:text-primary-400">
          {{ Math.round(progress) }}%
        </span>
      </div>
      <div class="h-2 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div
          class="h-full rounded-full bg-primary-600 transition-all duration-500 ease-out dark:bg-primary-500"
          :style="{ width: `${progress}%` }"
        />
      </div>
    </div>

    <!-- Steps -->
    <nav aria-label="Progress">
      <ol class="flex items-center justify-between">
        <li
          v-for="(step, index) in steps"
          :key="step.level"
          class="relative flex flex-1 flex-col items-center"
          :class="{ 'pr-8': index < steps.length - 1 }"
        >
          <!-- Connector Line -->
          <div
            v-if="index < steps.length - 1"
            class="absolute left-1/2 top-5 h-0.5 w-full"
            :class="{
              'bg-primary-600 dark:bg-primary-500': step.completed,
              'bg-gray-200 dark:bg-gray-700': !step.completed,
            }"
            aria-hidden="true"
          />

          <!-- Step Circle -->
          <button
            class="relative z-10 flex h-10 w-10 items-center justify-center rounded-full border-2 transition-all"
            :class="{
              'border-primary-600 bg-primary-600 text-white dark:border-primary-500 dark:bg-primary-500': getStepStatus(step) === 'completed',
              'border-primary-600 bg-white text-primary-600 dark:border-primary-500 dark:bg-gray-800 dark:text-primary-400': getStepStatus(step) === 'current',
              'border-gray-300 bg-white text-gray-400 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-500': getStepStatus(step) === 'upcoming',
            }"
            :aria-current="step.level === currentStep ? 'step' : undefined"
          >
            <Icon
              v-if="step.completed"
              name="heroicons:check"
              class="h-5 w-5"
              aria-hidden="true"
            />
            <span v-else class="text-sm font-semibold">{{ step.level }}</span>
          </button>

          <!-- Step Info -->
          <div class="mt-2 text-center">
            <p
              class="text-xs font-medium"
              :class="{
                'text-primary-600 dark:text-primary-400': getStepStatus(step) !== 'upcoming',
                'text-gray-500 dark:text-gray-400': getStepStatus(step) === 'upcoming',
              }"
            >
              {{ step.title }}
            </p>
          </div>
        </li>
      </ol>
    </nav>
  </div>
</template>