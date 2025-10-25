<script setup lang='ts'>
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, AreaChart, Area } from 'recharts'
const isClient = ref(false)
onMounted(() => { isClient.value = true })
const trendData = ref([
  { name: 'Пн', uptime: 99.7, alerts: 2 },
  { name: 'Вт', uptime: 99.9, alerts: 1 },
  { name: 'Ср', uptime: 99.8, alerts: 3 },
  { name: 'Чт', uptime: 99.95, alerts: 1 },
  { name: 'Пт', uptime: 99.92, alerts: 2 },
  { name: 'Сб', uptime: 99.96, alerts: 1 },
  { name: 'Вс', uptime: 99.94, alerts: 1 }
])
const efficiencyData = ref([
  { name: 'A', temp: 45.2, pressure: 151 },
  { name: 'B', temp: 52.1, pressure: 145 },
  { name: 'C', temp: 41.8, pressure: 140 }
])
</script>
<template>
  <div class="premium-card p-6 mt-6">
    <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">📉 Тренды за неделю</h3>
    <ClientOnly>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
          <div class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Uptime (%)</div>
          <ResponsiveContainer width="100%" :height="240">
            <AreaChart :data="trendData">
              <defs>
                <linearGradient id="colorUptime" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stop-color="#3b82f6" stop-opacity="0.5"/>
                  <stop offset="95%" stop-color="#3b82f6" stop-opacity="0"/>
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#e5e7eb" stroke-dasharray="3 3" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis :domain="[99.6, 100]" stroke="#9ca3af" />
              <Tooltip />
              <Area type="monotone" dataKey="uptime" stroke="#3b82f6" fill="url(#colorUptime)" :stroke-width="2" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div class="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
          <div class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Алерты (шт.)</div>
          <ResponsiveContainer width="100%" :height="240">
            <BarChart :data="trendData">
              <CartesianGrid stroke="#e5e7eb" stroke-dasharray="3 3" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis :allowDecimals="false" stroke="#9ca3af" />
              <Tooltip />
              <Bar dataKey="alerts" fill="#ef4444" :radius="[8,8,0,0]" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </ClientOnly>
  </div>
</template>