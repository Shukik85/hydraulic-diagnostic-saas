/**
 * vue-echarts.client.ts — Plugin для регистрации ECharts компонентов
 * 
 * Регистрирует <v-chart /> компонент глобально для использования
 * в diagnostic visualizations и dashboards
 */
import { defineNuxtPlugin } from '#app'
import VueECharts from 'vue-echarts'
import { use } from 'echarts/core'

// Import required ECharts components
import {
  CanvasRenderer
} from 'echarts/renderers'
import {
  LineChart,
  ScatterChart,
  GraphChart
} from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  ToolboxComponent,
  DataZoomComponent,
  MarkLineComponent,
  MarkPointComponent,
  MarkAreaComponent
} from 'echarts/components'

// Register ECharts components
use([
  CanvasRenderer,
  LineChart,
  ScatterChart,
  GraphChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  ToolboxComponent,
  DataZoomComponent,
  MarkLineComponent,
  MarkPointComponent,
  MarkAreaComponent
])

export default defineNuxtPlugin((nuxtApp) => {
  // Register v-chart component globally
  nuxtApp.vueApp.component('v-chart', VueECharts)
})
