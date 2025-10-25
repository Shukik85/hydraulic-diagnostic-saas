<script setup lang='ts'>
// Tiny sparklines for quick glance metrics
import { LineChart, Line, ResponsiveContainer } from 'recharts'

const sparkDataTemp = [41, 42, 44, 45, 43, 46, 45, 44, 45, 46]
const sparkDataPressure = [140, 142, 145, 147, 146, 148, 149, 150, 151, 150]
const sparkDataFlow = [68, 70, 72, 71, 73, 74, 75, 76, 75, 74]
const sparkDataVibration = [0.8, 0.9, 1.1, 1.0, 0.95, 1.2, 1.1, 1.0, 0.95, 1.0]
</script>
<template>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Температура</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">45.2°C</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="sparkDataTemp.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#ef4444" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Давление</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">151 бар</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="sparkDataPressure.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#3b82f6" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Расход</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">85.4 л/мин</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="sparkDataFlow.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#10b981" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Вибрация</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">1.0 мм/с</span>
      </div>
      <div class="h-10">
        <ResponsiveContainer width="100%" :height="40">
          <LineChart :data="sparkDataVibration.map((v, i) => ({ i, v }))">
            <Line type="monotone" dataKey="v" stroke="#a855f7" :strokeWidth="2" dot="false" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  </div>
</template>
