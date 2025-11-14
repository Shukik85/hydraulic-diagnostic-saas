<script setup lang="ts">
// Professional KPI card component with metallic industrial styling
interface Props {
  title: string;
  value: string | number;
  icon: string;
  color?: 'primary' | 'success' | 'warning' | 'error' | 'info' | 'steel';
  growth?: number;
  loadingState?: string | boolean;
  description?: string;
}

const props = withDefaults(defineProps<Props>(), {
  color: 'primary',
  loadingState: false,
});

// Type-safe color mapping - Industrial palette
type ColorKey = NonNullable<Props['color']>;

const getColorClasses = (color: ColorKey): string => {
  const colorMap: Record<ColorKey, string> = {
    primary: 'from-primary-600 to-primary-700 bg-primary-500/20 text-primary-300 border-primary-500/30',
    success: 'from-status-success to-status-success-dark bg-status-success/20 text-status-success-light border-status-success/30',
    warning: 'from-status-warning to-status-warning-dark bg-status-warning/20 text-status-warning-light border-status-warning/30',
    error: 'from-status-error to-status-error-dark bg-status-error/20 text-status-error-light border-status-error/30',
    info: 'from-status-info to-status-info-dark bg-status-info/20 text-status-info-light border-status-info/30',
    steel: 'from-steel-medium to-steel-light bg-steel-dark/20 text-steel-shine border-steel-medium/30',
  };

  return colorMap[color];
};

const formatGrowth = (value?: number): string => {
  if (value === undefined || value === null) return '';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}%`;
};

const getGrowthColor = (value?: number): string => {
  if (value === undefined || value === null) return 'text-text-muted';
  return value >= 0 ? 'text-status-success-light' : 'text-status-error-light';
};

const getGrowthIcon = (value?: number): string => {
  if (value === undefined || value === null) return 'heroicons:minus';
  return value >= 0 ? 'heroicons:arrow-trending-up' : 'heroicons:arrow-trending-down';
};

// Loading state management
const isLoading = computed(() => {
  return (
    props.loadingState === true || props.loadingState === 'loading' || props.loadingState === 'true'
  );
});
</script>

<template>
  <div class="card-metal p-6 hover:scale-[1.02] transition-all duration-300 group">
    <!-- Loading state -->
    <div v-if="isLoading" class="animate-pulse">
      <div class="flex items-center justify-between mb-4">
        <div class="w-16 h-4 bg-steel-medium rounded"></div>
        <div class="w-8 h-8 bg-steel-medium rounded-lg"></div>
      </div>
      <div class="w-20 h-8 bg-steel-medium rounded mb-2"></div>
      <div class="w-12 h-3 bg-steel-medium rounded"></div>
    </div>

    <!-- Content state -->
    <div v-else>
      <!-- Header -->
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm text-text-secondary font-semibold uppercase tracking-wide">
          {{ title }}
        </h3>
        <div
          :class="[
            'w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300',
            'group-hover:scale-110 group-hover:rotate-3',
            `bg-gradient-to-br ${getColorClasses(color || 'primary')}`,
          ]"
        >
          <Icon :name="icon" class="w-5 h-5 text-white" />
        </div>
      </div>

      <!-- Value -->
      <div class="mb-2">
        <div
          class="text-3xl font-bold text-text-primary group-hover:text-primary-300 transition-colors text-glow"
        >
          {{ typeof value === 'number' ? value.toLocaleString() : value }}
        </div>
      </div>

      <!-- Growth indicator -->
      <div v-if="growth !== undefined" class="flex items-center space-x-2">
        <Icon :name="getGrowthIcon(growth)" :class="['w-4 h-4', getGrowthColor(growth)]" />
        <span :class="['text-sm font-medium', getGrowthColor(growth)]">
          {{ formatGrowth(growth) }}
        </span>
        <span class="text-text-muted text-sm">vs прошлый период</span>
      </div>

      <!-- Description -->
      <div v-if="description" class="mt-3 text-sm text-text-secondary">
        {{ description }}
      </div>
    </div>
  </div>
</template>