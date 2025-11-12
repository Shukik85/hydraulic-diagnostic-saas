# Phase 2: Diagnostic Visualization - Complete Documentation

Все компоненты PHASE 2 восстановлены и закоммичены!

## 🎯 Обзор

Phase 2 включает полноценную систему диагностики с real-time мониторингом, GNN анализом и визуализацией архитектуры системы.

## 📦 Созданные компоненты (5 коммитов)

### 1. `plugins/vue-echarts.client.ts`

**Назначение:** Регистрация ECharts компонентов

**Возможности:**
- ✅ Глобальная регистрация `<v-chart />`
- ✅ LineChart, ScatterChart, GraphChart
- ✅ DataZoom, MarkLine, MarkArea
- ✅ Toolbox с export to image

---

### 2. `components/diagnostics/SensorChart.vue`

**Назначение:** Time-series графики для данных сенсоров

**Возможности:**
- ✅ Настраиваемый refresh interval (10s/30s/60s/Manual)
- ✅ Сохранение выбора в localStorage
- ✅ Expected range zones (зелёная зона)
- ✅ Anomaly markers (красные точки)
- ✅ Zoom & pan
- ✅ Export to image
- ✅ Статистика (Current/Min/Max/Avg)
- ✅ Цветовая индикация current value

**Props:**
```typescript
interface Props {
  sensorId: string
  sensorName: string
  sensorType: string
  unit?: string // default: 'bar'
  expectedRange?: { min: number; max: number }
  chartHeight?: string // default: '300px'
  timeRange?: number // minutes, default: 60
}
```

**Events:**
```typescript
emit('anomalyClick', data) // Клик на anomaly point
```

**Пример использования:**
```vue
<SensorChart
  sensor-id="sensor-123"
  sensor-name="Pressure Pump A"
  sensor-type="pressure"
  unit="bar"
  :expected-range="{ min: 50, max: 150 }"
  @anomaly-click="handleAnomalyClick"
/>
```

---

### 3. `components/diagnostics/GraphView.vue`

**Назначение:** Force-directed graph архитектуры системы

**Возможности:**
- ✅ Узлы = компоненты
- ✅ Рёбра = связи из adjacency matrix
- ✅ Цвет узла = anomaly score
  - Зелёный: score < 0.3
  - Жёлтый: 0.3 ≤ score < 0.7
  - Красный: score ≥ 0.7
- ✅ Размер узла = пропорционален anomaly score
- ✅ Interactive hover tooltips
- ✅ Zoom & pan
- ✅ Click to select component
- ✅ Focus on adjacency (подсветка связанных)
- ✅ Legend по типам компонентов
- ✅ Reset layout button

**Props:**
```typescript
interface Props {
  components: ComponentMetadata[]
  adjacencyMatrix: number[][]
  anomalyScores?: Record<string, number>
  graphHeight?: string // default: '500px'
}
```

**Events:**
```typescript
emit('componentSelect', component) // Клик на узел
```

**Пример использования:**
```vue
<GraphView
  :components="components"
  :adjacency-matrix="adjacencyMatrix"
  :anomaly-scores="{ 'comp-1': 0.8, 'comp-2': 0.2 }"
  @component-select="handleComponentSelect"
/>
```

---

### 4. `components/diagnostics/DiagnosticsDashboard.vue`

**Назначение:** Главный dashboard диагностики

**Возможности:**
- ✅ GraphView для архитектуры
- ✅ Grid из SensorChart (2 columns responsive)
- ✅ GNN Inference button
- ✅ GNN Results display:
  - System Health Score
  - Detected Anomalies count
  - Prediction Confidence
- ✅ Component Anomaly Scores (grid 4 columns)
- ✅ Recommendations panel (с priority)
- ✅ Export to CSV/PDF

**Props:**
```typescript
interface Props {
  equipmentId: string
  components: ComponentMetadata[]
  adjacencyMatrix: number[][]
  sensors: Array<{
    id: string
    name: string
    type: string
    unit: string
    expectedRange?: { min: number; max: number }
  }>
}
```

**API Integration:**
- `POST /api/gnn/infer` - запуск GNN анализа
- `GET /api/sensors/{id}/readings` - данные сенсоров

**Пример использования:**
```vue
<DiagnosticsDashboard
  equipment-id="eq-123"
  :components="components"
  :adjacency-matrix="adjacencyMatrix"
  :sensors="sensors"
/>
```

---

### 5. `pages/equipment/[id]/diagnostics.vue`

**Назначение:** Страница диагностики

**Возможности:**
- ✅ Breadcrumbs navigation
- ✅ Loading state
- ✅ Error state с retry
- ✅ No data state
- ✅ Интеграция с DiagnosticsDashboard
- ✅ SEO metadata

**Data Loading:**
1. `GET /api/equipment/{id}` - equipment details
2. `GET /api/metadata/systems?equipment_id={id}` - метаданные
3. `GET /api/sensor-mappings?equipment_id={id}` - sensor mappings

**Route:** `/equipment/:id/diagnostics`

---

## 🛠️ Технические детали

### ECharts Configuration

**Line Chart (для time-series):**
```typescript
{
  xAxis: { type: 'time' },
  yAxis: { type: 'value' },
  series: [{
    type: 'line',
    smooth: true,
    areaStyle: { /* gradient */ },
    markArea: { /* expected range */ }
  }]
}
```

**Graph Chart (для force-directed):**
```typescript
{
  series: [{
    type: 'graph',
    layout: 'force',
    force: {
      repulsion: 1000,
      edgeLength: 150,
      gravity: 0.1
    },
    emphasis: {
      focus: 'adjacency'
    }
  }]
}
```

### Refresh Intervals

**localStorage key:** `sensor_refresh_{sensorId}`

**Options:**
- 10s = 10000ms
- 30s = 30000ms (default)
- 1min = 60000ms
- Manual = 0

**Implementation:**
```typescript
let refreshTimer: ReturnType<typeof setInterval> | null = null

function onRefreshIntervalChange(interval: number) {
  localStorage.setItem(`sensor_refresh_${sensorId}`, interval.toString())
  
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
  
  if (interval > 0) {
    refreshTimer = setInterval(fetchData, interval)
  }
}
```

### Color Coding

**Anomaly Score Colors:**
- `score < 0.3`: Green (#22c55e)
- `0.3 ≤ score < 0.7`: Yellow (#eab308)
- `score ≥ 0.7`: Red (#ef4444)

**Status Badges:**
- `operational`: Green
- `warning`: Yellow
- `critical`: Red

---

## 📊 Performance

### Chart Optimization

**SensorChart:**
- Auto-resize on window resize
- Debounced refresh (prevent overlapping)
- Data point limit: 1000 points max
- Virtual rendering for large datasets

**GraphView:**
- Force-directed layout calculation offloaded to ECharts
- Node limit: 100 nodes recommended
- Edge limit: 500 edges recommended

### Memory Management

**Cleanup on unmount:**
```typescript
onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
  if (chartRef.value) chartRef.value.dispose()
})
```

---

## 🚀 Использование

### Quick Start

1. **Перейти на страницу diagnostics:**
   ```
   /equipment/{equipmentId}/diagnostics
   ```

2. **Запустить GNN Analysis:**
   - Нажать "Run GNN Analysis"
   - Подождать завершения анализа
   - Просмотреть results

3. **Настроить refresh interval:**
   - Выбрать интервал в dropdown
   - Настройка сохраняется для каждого сенсора

4. **Экспортировать данные:**
   - Export → Export to CSV
   - Файл скачается автоматически

### Integration Example

**Добавление ссылки в Equipment Detail:**

```vue
<!-- pages/equipment/[id].vue -->
<template>
  <div>
    <!-- ... existing tabs ... -->
    
    <!-- Add Diagnostics tab -->
    <UButton
      :to="`/equipment/${equipment.id}/diagnostics`"
      icon="i-heroicons-chart-bar"
    >
      View Diagnostics
    </UButton>
  </div>
</template>
```

---

## ✅ Testing Checklist

- [ ] SensorChart отображает данные
- [ ] Refresh interval работает
- [ ] Expected range zone отображается
- [ ] Anomaly markers кликабельны
- [ ] GraphView отображает архитектуру
- [ ] Цвета узлов соответствуют anomaly scores
- [ ] GNN Analysis запускается
- [ ] GNN Results отображаются
- [ ] Recommendations panel работает
- [ ] Export to CSV работает
- [ ] Loading states отображаются
- [ ] Error handling работает

---

## 📝 Next Steps

### PHASE 3: Real-time Dashboard (после MVP)

1. **Home Dashboard Overview**
   - Общая статистика по всем системам
   - Recent alerts list
   - 24h activity charts

2. **WebSocket Integration**
   - Real-time sensor updates
   - Live anomaly notifications
   - Auto-refresh dashboards

3. **Mobile Responsive**
   - Adaptive layouts
   - Touch gestures
   - PWA support

---

## 👥 Support

Все компоненты PHASE 2 готовы к использованию! 🎉

Если есть вопросы или нужны доработки - пиши! 🚀
