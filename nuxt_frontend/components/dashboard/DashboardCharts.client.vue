<script setup lang='ts'>
const trendData = ref([
  { name: 'Пн', uptime: 99.7, alerts: 2 },
  { name: 'Вт', uptime: 99.9, alerts: 1 },
  { name: 'Ср', uptime: 99.8, alerts: 3 },
  { name: 'Чт', uptime: 99.95, alerts: 1 },
  { name: 'Пт', uptime: 99.92, alerts: 2 },
  { name: 'Сб', uptime: 99.96, alerts: 1 },
  { name: 'Вс', uptime: 99.94, alerts: 1 }
])

// Simple canvas-based charts
const drawChart = (canvas: HTMLCanvasElement, data: number[], type: 'area' | 'bar', color: string) => {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  const width = canvas.width
  const height = canvas.height
  const padding = 20
  
  ctx.clearRect(0, 0, width, height)
  
  if (type === 'area') {
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1
    
    ctx.beginPath()
    data.forEach((value, index) => {
      const x = padding + (index / (data.length - 1)) * (width - 2 * padding)
      const y = height - padding - ((value - min) / range) * (height - 2 * padding)
      if (index === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    
    // Area fill
    ctx.lineTo(width - padding, height - padding)
    ctx.lineTo(padding, height - padding)
    ctx.closePath()
    
    const gradient = ctx.createLinearGradient(0, 0, 0, height)
    gradient.addColorStop(0, color + '80')
    gradient.addColorStop(1, color + '20')
    ctx.fillStyle = gradient
    ctx.fill()
    
    // Line
    ctx.beginPath()
    data.forEach((value, index) => {
      const x = padding + (index / (data.length - 1)) * (width - 2 * padding)
      const y = height - padding - ((value - min) / range) * (height - 2 * padding)
      if (index === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.strokeStyle = color
    ctx.lineWidth = 3
    ctx.stroke()
  } else if (type === 'bar') {
    const max = Math.max(...data)
    const barWidth = (width - 2 * padding) / data.length * 0.6
    const barSpacing = (width - 2 * padding) / data.length
    
    data.forEach((value, index) => {
      const x = padding + index * barSpacing + (barSpacing - barWidth) / 2
      const barHeight = (value / max) * (height - 2 * padding)
      const y = height - padding - barHeight
      
      ctx.fillStyle = color
      ctx.fillRect(x, y, barWidth, barHeight)
    })
  }
}

const uptimeCanvasRef = ref<HTMLCanvasElement>()
const alertsCanvasRef = ref<HTMLCanvasElement>()

onMounted(() => {
  if (uptimeCanvasRef.value) {
    uptimeCanvasRef.value.width = 400
    uptimeCanvasRef.value.height = 240
    drawChart(uptimeCanvasRef.value, trendData.value.map(d => d.uptime), 'area', '#3b82f6')
  }
  
  if (alertsCanvasRef.value) {
    alertsCanvasRef.value.width = 400
    alertsCanvasRef.value.height = 240
    drawChart(alertsCanvasRef.value, trendData.value.map(d => d.alerts), 'bar', '#ef4444')
  }
})
</script>

<template>
  <div class="premium-card p-6 mt-6">
    <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">📊 Тренды за неделю</h3>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Uptime chart -->
      <div class="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
        <div class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Время работы (%)</div>
        <div class="h-60">
          <canvas ref="uptimeCanvasRef" class="w-full h-full"></canvas>
        </div>
      </div>

      <!-- Alerts chart -->
      <div class="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
        <div class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Алерты (шт.)</div>
        <div class="h-60">
          <canvas ref="alertsCanvasRef" class="w-full h-full"></canvas>
        </div>
      </div>
    </div>
  </div>
</template>