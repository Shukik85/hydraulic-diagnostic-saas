<script setup lang="ts">
interface Props {
  title: string
  description?: string
  icon?: string
  iconColor?: 'primary' | 'success' | 'warning' | 'error' | 'info'
  actionText?: string
  actionHref?: string
  badge?: string
  badgeColor?: 'primary' | 'success' | 'warning' | 'error' | 'info'
}

const props = withDefaults(defineProps<Props>(), {
  iconColor: 'primary',
  badgeColor: 'primary',
})

type ColorKey = NonNullable<Props['iconColor']>

const getIconColorClass = (color: ColorKey): string => {
  const colorMap: Record<ColorKey, string> = {
    primary: 'text-primary-500',
    success: 'text-success-500',
    warning: 'text-warning-500',
    error: 'text-error-500',
    info: 'text-primary-400',
  }
  return colorMap[color]
}

const getBadgeColorClass = (color: ColorKey): string => {
  const colorMap: Record<ColorKey, string> = {
    primary: 'bg-primary-500/10 text-primary-400 border border-primary-500/30',
    success: 'bg-success-500/10 text-success-400 border border-success-500/30',
    warning: 'bg-warning-500/10 text-warning-400 border border-warning-500/30',
    error: 'bg-error-500/10 text-error-400 border border-error-500/30',
    info: 'bg-primary-500/10 text-primary-300 border border-primary-500/30',
  }
  return colorMap[color]
}
</script>

<template>
  <div class="flex items-center justify-between mb-6 pb-4 border-b border-steel-medium">
    <div class="flex-1">
      <div class="flex items-center space-x-4 mb-2">
        <!-- Icon -->
        <Icon
          v-if="icon"
          :name="icon"
          :class="`w-6 h-6 ${getIconColorClass(iconColor || 'primary')}`"
        />

        <!-- Title -->
        <h2 class="text-2xl font-bold text-text-primary tracking-tight">
          {{ title }}
        </h2>

        <!-- Badge -->
        <span
          v-if="badge"
          :class="`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wide ${getBadgeColorClass(badgeColor || 'primary')}`"
        >
          {{ badge }}
        </span>
      </div>

      <!-- Description -->
      <p v-if="description" class="text-sm text-text-secondary">
        {{ description }}
      </p>
    </div>

    <!-- Action button -->
    <div v-if="actionText && actionHref">
      <NuxtLink
        :to="actionHref"
        class="inline-flex items-center px-4 py-2 text-sm font-bold rounded-lg text-white bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 shadow-lg shadow-primary-500/30 hover:shadow-xl hover:shadow-primary-500/40 transition-all duration-200 hover:scale-105"
      >
        {{ actionText }}
        <Icon name="heroicons:arrow-right" class="w-4 h-4 ml-2" />
      </NuxtLink>
    </div>
  </div>
</template>