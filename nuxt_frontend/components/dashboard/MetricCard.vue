<template>
    <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div class="flex items-center">
            <div class="flex-shrink-0">
                <div class="w-10 h-10 rounded-full flex items-center justify-center" :class="iconBgClass">
                    <Icon :name="icon" class="w-5 h-5" :class="iconClass" />
                </div>
            </div>

            <div class="ml-4 flex-1">
                <div class="text-sm font-medium text-gray-600">{{ title }}</div>
                <div class="text-2xl font-bold text-gray-900">
                    {{ formatValue(value) }}
                    <span v-if="unit" class="text-sm font-normal text-gray-500 ml-1">{{ unit }}</span>
                </div>

                <div v-if="trend" class="flex items-center mt-1">
                    <Icon :name="trendIcon" class="w-4 h-4 mr-1" :class="trendClass" />
                    <span class="text-xs font-medium" :class="trendClass">
                        {{ trend }}
                    </span>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
interface Props {
    title: string
    value: number | string
    unit?: string
    trend?: string
    icon?: string
    type?: 'success' | 'warning' | 'error' | 'info' | 'default'
}

const props = withDefaults(defineProps<Props>(), {
    icon: 'heroicons:chart-bar',
    type: 'default'
})

const formatValue = (val: number | string) => {
    if (typeof val === 'number') {
        return val.toLocaleString('ru-RU', { maximumFractionDigits: 1 })
    }
    return val
}

const iconBgClass = computed(() => {
    const classes = {
        success: 'bg-green-100',
        warning: 'bg-yellow-100',
        error: 'bg-red-100',
        info: 'bg-blue-100',
        default: 'bg-gray-100'
    }
    return classes[props.type]
})

const iconClass = computed(() => {
    const classes = {
        success: 'text-green-600',
        warning: 'text-yellow-600',
        error: 'text-red-600',
        info: 'text-blue-600',
        default: 'text-gray-600'
    }
    return classes[props.type]
})

const trendIcon = computed(() => {
    if (!props.trend) return ''

    if (props.trend.startsWith('+') || !props.trend.startsWith('-')) {
        return 'heroicons:arrow-trending-up'
    } else {
        return 'heroicons:arrow-trending-down'
    }
})

const trendClass = computed(() => {
    if (!props.trend) return ''

    if (props.trend.startsWith('+') || !props.trend.startsWith('-')) {
        return 'text-green-600'
    } else {
        return 'text-red-600'
    }
})
</script>