<script setup lang='ts'>
// Sparklines fed by real-time-like demo data from external market (BTC volatility mapped)
import { LineChart, Line, ResponsiveContainer } from 'recharts'

const props = defineProps<{ temp: number[], pressure: number[], flow: number[], vibration: number[] }>()
</script>
<template>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Температура</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ temp[temp.length-1] }}°C</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="temp.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#ef4444" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Давление</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ pressure[pressure.length-1] }} бар</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="pressure.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#3b82f6" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Расход</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ flow[flow.length-1] }} л/мин</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="flow.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#10b981" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Вибрация</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ vibration[vibration.length-1] }} мм/с</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="vibration.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#a855f7" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  </div>
</template>
