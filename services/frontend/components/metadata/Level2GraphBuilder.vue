<!-- components/metadata/Level2GraphBuilder.vue -->
<template>
  <div class="level-2">
    <h2 class="text-xl font-semibold mb-4">2. Архитектура гидросистемы</h2>

    <p class="text-gray-600 mb-6">
      Постройте схему гидравлической системы, добавляя компоненты и связи между ними.
      Это поможет модели понять физическую топологию для анализа каскадных отказов.
    </p>

    <div class="graph-builder">
      <!-- Toolbar -->
      <div class="toolbar">
        <h3 class="toolbar-title">Палитра компонентов</h3>

        <div class="component-palette">
          <button v-for="compType in componentTypes" :key="compType.type" @click="selectComponentType(compType.type)"
            :class="['palette-btn', { 'selected': selectedType === compType.type }]" :title="compType.description">
            <span class="component-icon">{{ compType.icon }}</span>
            <span class="component-label">{{ compType.label }}</span>
          </button>
        </div>

        <div class="toolbar-section">
          <h4 class="text-sm font-medium mb-2">Добавленные компоненты ({{ store.componentsCount }})</h4>
          <div class="component-list">
            <div v-for="comp in store.wizardState.system.components" :key="comp.id"
              :class="['component-item', { 'selected': selectedComponent === comp.id }]"
              @click="selectComponent(comp.id)">
              <span class="component-icon-sm">{{ getComponentIcon(comp.component_type) }}</span>
              <span class="component-name">{{ comp.id }}</span>
              <button @click.stop="removeComponent(comp.id)" class="remove-btn" title="Удалить">×</button>
            </div>
          </div>
        </div>

        <div class="toolbar-actions">
          <button @click="clearAll" class="btn-secondary btn-sm">Очистить всё</button>
          <button @click="autoLayout" class="btn-secondary btn-sm">Авто-расстановка</button>
        </div>
      </div>

      <!-- Canvas -->
      <div class="canvas-container">
        <div class="canvas-header">
          <h3 class="text-sm font-medium">Схема системы</h3>
          <div class="canvas-controls">
            <button @click="zoomIn" class="control-btn" title="Увеличить">+</button>
            <span class="zoom-level">{{ Math.round(zoom * 100) }}%</span>
            <button @click="zoomOut" class="control-btn" title="Уменьшить">−</button>
            <button @click="resetView" class="control-btn" title="Сброс">⟲</button>
          </div>
        </div>

        <div ref="canvasContainer" class="canvas" @click="onCanvasClick" @mousemove="onCanvasMouseMove"
          @mouseup="onCanvasMouseUp">
          <svg ref="svgCanvas" :width="canvasWidth" :height="canvasHeight"
            :viewBox="`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`" class="canvas-svg">
            <!-- Grid -->
            <defs>
              <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e5e7eb" stroke-width="0.5" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />

            <!-- Connections (edges) -->
            <g class="connections-layer">
              <g v-for="(comp, i) in store.wizardState.system.components" :key="`conn-${comp.id}`">
                <line v-for="targetId in comp.connected_to" :key="`${comp.id}-${targetId}`" :x1="comp.position?.x || 0"
                  :y1="comp.position?.y || 0" :x2="getComponentPosition(targetId).x"
                  :y2="getComponentPosition(targetId).y" :stroke="getConnectionColor(comp.connection_types[targetId])"
                  stroke-width="2" stroke-dasharray="5,5" class="connection-line" />
              </g>
            </g>

            <!-- Components (nodes) -->
            <g class="components-layer">
              <g v-for="comp in store.wizardState.system.components" :key="comp.id"
                :transform="`translate(${comp.position?.x || 0}, ${comp.position?.y || 0})`"
                @mousedown="onComponentMouseDown($event, comp.id)" @click.stop="selectComponent(comp.id)"
                :class="['component-node', { 'selected': selectedComponent === comp.id }]">
                <!-- Component shape -->
                <rect x="-40" y="-30" width="80" height="60" :fill="getComponentColor(comp.component_type)"
                  stroke="#374151" stroke-width="2" rx="8" class="component-shape" />

                <!-- Component icon -->
                <text x="0" y="-5" text-anchor="middle" font-size="24" class="component-icon-text">
                  {{ getComponentIcon(comp.component_type) }}
                </text>

                <!-- Component label -->
                <text x="0" y="20" text-anchor="middle" font-size="10" fill="#1f2937" class="component-label-text">
                  {{ comp.id }}
                </text>

                <!-- Connection points -->
                <circle v-if="selectedComponent === comp.id" cx="40" cy="0" r="6" fill="#3b82f6"
                  class="connection-point" @mousedown.stop="startConnection(comp.id)" />
              </g>
            </g>

            <!-- Drawing connection line -->
            <line v-if="drawingConnection && connectionStart" :x1="connectionStart.x" :y1="connectionStart.y"
              :x2="mousePosition.x" :y2="mousePosition.y" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,5"
              class="drawing-line" />
          </svg>
        </div>

        <!-- Canvas hints -->
        <div class="canvas-hints">
          <p v-if="selectedType" class="hint">
            💡 Кликните на холсте, чтобы добавить {{ getComponentLabel(selectedType) }}
          </p>
          <p v-else-if="selectedComponent" class="hint">
            💡 Кликните на синюю точку справа, чтобы создать связь с другим компонентом
          </p>
          <p v-else class="hint">
            💡 Выберите компонент из палитры слева или кликните на существующий компонент
          </p>
        </div>
      </div>
    </div>

    <!-- Connection Type Modal -->
    <div v-if="showConnectionModal" class="modal-overlay" @click="cancelConnection">
      <div class="modal" @click.stop>
        <h3 class="modal-title">Тип соединения</h3>
        <p class="modal-description">
          Выберите тип гидравлической линии между компонентами:
        </p>

        <div class="connection-types">
          <button @click="completeConnection('pressure_line')" class="connection-type-btn">
            <span class="type-icon" style="background: #ef4444"></span>
            <div>
              <div class="type-label">Напорная линия</div>
              <div class="type-description">Высокое давление от насоса</div>
            </div>
          </button>

          <button @click="completeConnection('return_line')" class="connection-type-btn">
            <span class="type-icon" style="background: #3b82f6"></span>
            <div>
              <div class="type-label">Обратная линия</div>
              <div class="type-description">Возврат жидкости в резервуар</div>
            </div>
          </button>

          <button @click="completeConnection('pilot_line')" class="connection-type-btn">
            <span class="type-icon" style="background: #f59e0b"></span>
            <div>
              <div class="type-label">Пилотная линия</div>
              <div class="type-description">Управляющий сигнал</div>
            </div>
          </button>
        </div>

        <button @click="cancelConnection" class="btn-secondary btn-sm mt-4">Отмена</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { useMetadataStore } from '~/stores/metadata';
import type { ComponentType } from '~/types/metadata';

const store = useMetadataStore();

const componentTypes = [
  { type: 'pump' as ComponentType, icon: '⚙️', label: 'Насос', description: 'Гидравлический насос' },
  { type: 'motor' as ComponentType, icon: '🔄', label: 'Мотор', description: 'Гидромотор' },
  { type: 'cylinder' as ComponentType, icon: '⬌', label: 'Цилиндр', description: 'Гидроцилиндр' },
  { type: 'valve' as ComponentType, icon: '⬥', label: 'Клапан', description: 'Гидравлический клапан' },
  { type: 'filter' as ComponentType, icon: '◈', label: 'Фильтр', description: 'Масляный фильтр' },
  { type: 'accumulator' as ComponentType, icon: '⬢', label: 'Аккумулятор', description: 'Гидроаккумулятор' },
];

// Canvas state
const canvasContainer = ref<HTMLDivElement>();
const svgCanvas = ref<SVGSVGElement>();
const canvasWidth = ref(800);
const canvasHeight = ref(600);
const zoom = ref(1);
const viewBox = ref({ x: 0, y: 0, width: 800, height: 600 });

// Selection state
const selectedType = ref<ComponentType | null>(null);
const selectedComponent = ref<string | null>(null);

// Drawing state
const drawingConnection = ref(false);
const connectionStart = ref<{ id: string; x: number; y: number } | null>(null);
const connectionEnd = ref<string | null>(null);
const mousePosition = ref({ x: 0, y: 0 });
const showConnectionModal = ref(false);

// Dragging state
const draggingComponent = ref<string | null>(null);
const dragOffset = ref({ x: 0, y: 0 });

let componentCounter = 0;

function selectComponentType(type: ComponentType) {
  selectedType.value = selectedType.value === type ? null : type;
  selectedComponent.value = null;
}

function selectComponent(id: string) {
  selectedComponent.value = selectedComponent.value === id ? null : id;
  selectedType.value = null;
}

function onCanvasClick(event: MouseEvent) {
  if (!selectedType.value) return;

  const svg = svgCanvas.value;
  if (!svg) return;

  const rect = svg.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (viewBox.value.width / rect.width) + viewBox.value.x;
  const y = (event.clientY - rect.top) * (viewBox.value.height / rect.height) + viewBox.value.y;

  addComponent(selectedType.value, x, y);
  selectedType.value = null;
}

function addComponent(type: ComponentType, x: number, y: number) {
  componentCounter++;
  const id = `${type}_${componentCounter}`;

  store.addComponent({
    id,
    component_type: type,
    normal_ranges: {
      pressure: undefined,
      temperature: undefined,
      flow_rate: undefined,
      vibration: undefined
    },
    connected_to: [],
    connection_types: {},
    confidence_scores: {},
    position: { x, y }
  } as any);
}

function removeComponent(id: string) {
  store.removeComponent(id);
  if (selectedComponent.value === id) {
    selectedComponent.value = null;
  }
}

function clearAll() {
  if (confirm('Удалить все компоненты?')) {
    store.wizardState.system.components = [];
    store.wizardState.system.adjacency_matrix = [];
    componentCounter = 0;
  }
}

function autoLayout() {
  const components = store.wizardState.system.components || [];
  const centerX = viewBox.value.width / 2;
  const centerY = viewBox.value.height / 2;
  const radius = 200;

  components.forEach((comp, i) => {
    const angle = (i / components.length) * 2 * Math.PI;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);

    comp.position = { x, y };
  });
}

// Connection drawing
function startConnection(fromId: string) {
  const comp = store.wizardState.system.components?.find(c => c.id === fromId);
  if (!comp || !comp.position) return;

  drawingConnection.value = true;
  connectionStart.value = { id: fromId, x: comp.position.x + 40, y: comp.position.y };
}

function onCanvasMouseMove(event: MouseEvent) {
  const svg = svgCanvas.value;
  if (!svg) return;

  const rect = svg.getBoundingClientRect();
  mousePosition.value = {
    x: (event.clientX - rect.left) * (viewBox.value.width / rect.width) + viewBox.value.x,
    y: (event.clientY - rect.top) * (viewBox.value.height / rect.height) + viewBox.value.y
  };

  // Dragging component
  if (draggingComponent.value) {
    const comp = store.wizardState.system.components?.find(c => c.id === draggingComponent.value);
    if (comp) {
      comp.position = {
        x: mousePosition.value.x - dragOffset.value.x,
        y: mousePosition.value.y - dragOffset.value.y
      };
    }
  }
}

function onCanvasMouseUp(event: MouseEvent) {
  if (drawingConnection.value) {
    // Проверяем, на каком компоненте отпустили мышь
    const target = (event.target as SVGElement).closest('.component-node');
    if (target) {
      const targetId = store.wizardState.system.components?.find(
        c => c.position &&
          Math.abs(c.position.x - mousePosition.value.x) < 50 &&
          Math.abs(c.position.y - mousePosition.value.y) < 50
      )?.id;

      if (targetId && targetId !== connectionStart.value?.id) {
        connectionEnd.value = targetId;
        showConnectionModal.value = true;
        return;
      }
    }

    drawingConnection.value = false;
    connectionStart.value = null;
  }

  draggingComponent.value = null;
}

function completeConnection(type: 'pressure_line' | 'return_line' | 'pilot_line') {
  if (connectionStart.value && connectionEnd.value) {
    store.addConnection(connectionStart.value.id, connectionEnd.value, type);
  }

  drawingConnection.value = false;
  connectionStart.value = null;
  connectionEnd.value = null;
  showConnectionModal.value = false;
}

function cancelConnection() {
  drawingConnection.value = false;
  connectionStart.value = null;
  connectionEnd.value = null;
  showConnectionModal.value = false;
}

function onComponentMouseDown(event: MouseEvent, id: string) {
  event.stopPropagation();
  draggingComponent.value = id;

  const comp = store.wizardState.system.components?.find(c => c.id === id);
  if (comp && comp.position) {
    dragOffset.value = {
      x: mousePosition.value.x - comp.position.x,
      y: mousePosition.value.y - comp.position.y
    };
  }
}

// Helpers
function getComponentIcon(type: ComponentType): string {
  return componentTypes.find(ct => ct.type === type)?.icon || '?';
}

function getComponentLabel(type: ComponentType): string {
  return componentTypes.find(ct => ct.type === type)?.label || type;
}

function getComponentColor(type: ComponentType): string {
  const colors: Record<ComponentType, string> = {
    pump: '#fef3c7',
    motor: '#dbeafe',
    cylinder: '#e0e7ff',
    valve: '#fce7f3',
    filter: '#d1fae5',
    accumulator: '#fed7aa'
  };
  return colors[type] || '#f3f4f6';
}

function getConnectionColor(type: 'pressure_line' | 'return_line' | 'pilot_line' | undefined): string {
  const colors = {
    pressure_line: '#ef4444',
    return_line: '#3b82f6',
    pilot_line: '#f59e0b'
  };
  return type ? colors[type] : '#9ca3af';
}

function getComponentPosition(id: string): { x: number; y: number } {
  const comp = store.wizardState.system.components?.find(c => c.id === id);
  return comp?.position || { x: 0, y: 0 };
}

// Zoom controls
function zoomIn() {
  zoom.value = Math.min(zoom.value * 1.2, 3);
  updateViewBox();
}

function zoomOut() {
  zoom.value = Math.max(zoom.value / 1.2, 0.5);
  updateViewBox();
}

function resetView() {
  zoom.value = 1;
  viewBox.value = { x: 0, y: 0, width: 800, height: 600 };
}

function updateViewBox() {
  const newWidth = 800 / zoom.value;
  const newHeight = 600 / zoom.value;
  viewBox.value = {
    x: viewBox.value.x,
    y: viewBox.value.y,
    width: newWidth,
    height: newHeight
  };
}

onMounted(() => {
  if (canvasContainer.value) {
    canvasWidth.value = canvasContainer.value.clientWidth;
    canvasHeight.value = canvasContainer.value.clientHeight;
  }
});
</script>

<style scoped>
.level-2 {
  padding: 1rem;
}

.graph-builder {
  display: grid;
  grid-template-columns: 250px 1fr;
  gap: 1rem;
  height: 700px;
}

.toolbar {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.toolbar-title {
  font-weight: 600;
  font-size: 0.875rem;
  color: #374151;
}

.component-palette {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.palette-btn {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  border: 2px solid #e5e7eb;
  border-radius: 0.5rem;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.palette-btn:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.palette-btn.selected {
  border-color: #3b82f6;
  background: #3b82f6;
  color: white;
}

.component-icon {
  font-size: 1.5rem;
}

.component-label {
  font-size: 0.875rem;
  font-weight: 500;
}

.toolbar-section {
  border-top: 1px solid #e5e7eb;
  padding-top: 1rem;
}

.component-list {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-height: 200px;
  overflow-y: auto;
}

.component-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  border-radius: 0.375rem;
  cursor: pointer;
  transition: background 0.2s;
}

.component-item:hover {
  background: #f3f4f6;
}

.component-item.selected {
  background: #dbeafe;
}

.component-icon-sm {
  font-size: 1rem;
}

.component-name {
  flex: 1;
  font-size: 0.75rem;
  color: #4b5563;
}

.remove-btn {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #ef4444;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  transition: background 0.2s;
}

.remove-btn:hover {
  background: #dc2626;
}

.toolbar-actions {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: auto;
}

.canvas-container {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  display: flex;
  flex-direction: column;
}

.canvas-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #e5e7eb;
}

.canvas-controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.control-btn {
  width: 28px;
  height: 28px;
  border: 1px solid #d1d5db;
  border-radius: 0.25rem;
  background: white;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;
}

.control-btn:hover {
  background: #f3f4f6;
  border-color: #9ca3af;
}

.zoom-level {
  font-size: 0.75rem;
  color: #6b7280;
  min-width: 50px;
  text-align: center;
}

.canvas {
  flex: 1;
  position: relative;
  overflow: hidden;
  background: #fafafa;
}

.canvas-svg {
  width: 100%;
  height: 100%;
  cursor: crosshair;
}

.component-node {
  cursor: move;
  transition: transform 0.1s;
}

.component-node:hover .component-shape {
  stroke: #3b82f6;
  stroke-width: 3;
}

.component-node.selected .component-shape {
  stroke: #3b82f6;
  stroke-width: 3;
  filter: drop-shadow(0 0 8px rgba(59, 130, 246, 0.4));
}

.connection-line {
  pointer-events: none;
}

.drawing-line {
  pointer-events: none;
}

.connection-point {
  cursor: pointer;
}

.connection-point:hover {
  r: 8;
}

.canvas-hints {
  padding: 0.75rem 1rem;
  border-top: 1px solid #e5e7eb;
  background: #f9fafb;
}

.hint {
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0;
}

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
}

.modal {
  background: white;
  border-radius: 0.75rem;
  padding: 1.5rem;
  max-width: 400px;
  width: 90%;
}

.modal-title {
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.modal-description {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 1rem;
}

.connection-types {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.connection-type-btn {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 0.5rem;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.connection-type-btn:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.type-icon {
  width: 20px;
  height: 20px;
  border-radius: 4px;
}

.type-label {
  font-weight: 500;
  font-size: 0.875rem;
}

.type-description {
  font-size: 0.75rem;
  color: #6b7280;
}

.btn-secondary {
  padding: 0.5rem 1rem;
  background: #f3f4f6;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s;
}

.btn-secondary:hover {
  background: #e5e7eb;
}

.btn-sm {
  font-size: 0.875rem;
  padding: 0.5rem 0.75rem;
}
</style>
