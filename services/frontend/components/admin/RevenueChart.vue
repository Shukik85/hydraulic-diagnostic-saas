<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import VChart from 'vue-echarts';
import type { EChartsOption } from 'echarts';
import type { RevenuePoint } from '~/types';

interface Props {
  data: RevenuePoint[];
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
});

const { t } = useI18n();
const { formatCurrency } = useFormatting();

const chartOption = computed<EChartsOption>(() => ({
  tooltip: {
    trigger: 'axis',
    formatter: (params: any) => {
      const point = params[0];
      return `${point.name}<br/>${formatCurrency(point.value)}`;
    },
  },
  xAxis: {
    type: 'category',
    data: props.data.map((p) => p.date),
    axisLabel: {
      color: '#666',
    },
  },
  yAxis: {
    type: 'value',
    axisLabel: {
      color: '#666',
      formatter: (value: number) => formatCurrency(value, 'USD', 'en-US'),
    },
  },
  series: [
    {
      name: 'Revenue',
      type: 'line',
      data: props.data.map((p) => p.value),
      smooth: true,
      lineStyle: {
        color: '#14b8a6',
        width: 3,
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(20, 184, 166, 0.3)' },
            { offset: 1, color: 'rgba(20, 184, 166, 0.05)' },
          ],
        },
      },
      emphasis: {
        focus: 'series',
      },
    },
  ],
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true,
  },
}));
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <div class="mb-4 flex items-center justify-between">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {{ t('admin.revenueChart') }}
      </h3>
      <Icon name="heroicons:chart-bar" class="h-5 w-5 text-gray-400" aria-hidden="true" />
    </div>

    <div v-if="loading" class="flex h-80 items-center justify-center">
      <div class="h-8 w-8 animate-spin rounded-full border-4 border-primary-500 border-t-transparent" />
    </div>

    <VChart v-else :option="chartOption" class="h-80 w-full" autoresize />
  </div>
</template>
