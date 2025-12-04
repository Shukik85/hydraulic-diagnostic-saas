# Wizard GNN Integration

Онбординг гидравлических систем для GNN Service через пошаговый wizard.

## 📋 Структура Wizard

### Steps Overview

| Step | Component | Описание | Валидация |
|------|-----------|----------|----------|
| **Step 0** | `EquipmentInfo.vue` | Базовая информация об оборудовании | ID (1-100), Name (1-255), Hours ≥0 |
| **Step 1** | `SchemaUpload.vue` | Загрузка P&ID схемы (CSV/JSON) | *TODO: не реализован* |
| **Step 2** | `ComponentsEditor.vue` | Настройка компонентов | Min 2, уникальные ID, sensors ≥1 |
| **Step 3** | `TopologyEditor.vue` | Связи между компонентами | Min 1, no self-loops, components exist |
| **Step 4** | `ReviewSubmit.vue` | Проверка и отправка | Full topology validation |

## 🏗️ Архитектура

### Типы (types/gnn.ts)

```typescript
// Component types
enum ComponentType {
  HYDRAULIC_PUMP, GEAR_PUMP, PISTON_PUMP,
  HYDRAULIC_VALVE, DIRECTIONAL_VALVE,
  HYDRAULIC_CYLINDER, HYDRAULIC_MOTOR,
  // ... etc
}

// Edge types (connections)
enum EdgeType {
  HYDRAULIC_LINE, HIGH_PRESSURE_HOSE,
  LOW_PRESSURE_RETURN, PILOT_LINE,
  // ... etc
}

// Main topology interface
interface GraphTopology {
  equipmentId: string;
  equipmentName: string;
  components: Component[];
  edges: Edge[];
  topologyVersion?: string;
}
```

### Composables

#### `useTopologyValidation.ts`

```typescript
const { validateTopology, validateComponents, validateEdges } = useTopologyValidation();

// Validate full topology
const errors = validateTopology(topology);
// Returns: ValidationError[] = [{ field, message }, ...]
```

**Validation Rules:**
- Equipment ID: 1-100 chars, alphanumeric
- Components: min 2, unique IDs, at least 1 sensor each
- Edges: min 1, no self-loops, source/target must exist
- Ranges: pressure 0-1000 bar, flow 0-1000 L/min, diameter 0-500 mm

#### `useTopology.ts`

```typescript
const { submitTopology, loading, error } = useTopology();

await submitTopology(topology);
// POST /api/v1/topology
// Response: { topologyId, equipmentId, componentsCount, edgesCount }
```

## 🚀 Использование

### Пример: Главная страница Wizard

```vue
<script setup lang="ts">
import { ref } from 'vue';
import type { GraphTopology } from '~/types/gnn';
import EquipmentInfo from '~/components/wizard/steps/EquipmentInfo.vue';
import ComponentsEditor from '~/components/wizard/steps/ComponentsEditor.vue';
import TopologyEditor from '~/components/wizard/steps/TopologyEditor.vue';
import ReviewSubmit from '~/components/wizard/steps/ReviewSubmit.vue';

const currentStep = ref(0);
const topology = ref<GraphTopology>({
  equipmentId: '',
  equipmentName: '',
  components: [],
  edges: [],
  topologyVersion: 'v1.0',
});

const stepsValid = ref([false, false, false, false]);

const handleValidationChange = (step: number, isValid: boolean) => {
  stepsValid.value[step] = isValid;
};

const nextStep = () => {
  if (stepsValid.value[currentStep.value]) {
    currentStep.value++;
  }
};

const prevStep = () => {
  currentStep.value--;
};
</script>

<template>
  <div class="max-w-4xl mx-auto p-6">
    <ProgressIndicator :current="currentStep" :total="4" />

    <!-- Step 0: Equipment -->
    <EquipmentInfo
      v-if="currentStep === 0"
      v-model="topology"
      @validation-change="(valid) => handleValidationChange(0, valid)"
    />

    <!-- Step 2: Components -->
    <ComponentsEditor
      v-else-if="currentStep === 2"
      v-model="topology.components"
      @validation-change="(valid) => handleValidationChange(2, valid)"
    />

    <!-- Step 3: Edges -->
    <TopologyEditor
      v-else-if="currentStep === 3"
      v-model="topology.edges"
      :components="topology.components"
      @validation-change="(valid) => handleValidationChange(3, valid)"
    />

    <!-- Step 4: Review -->
    <ReviewSubmit
      v-else-if="currentStep === 4"
      :topology="topology"
      @submit-success="(res) => console.log('Success:', res)"
      @submit-error="(err) => console.error('Error:', err)"
    />

    <!-- Navigation -->
    <div class="flex justify-between mt-8">
      <Button
        variant="outline"
        :disabled="currentStep === 0"
        @click="prevStep"
      >
        Previous
      </Button>
      <Button
        v-if="currentStep < 4"
        :disabled="!stepsValid[currentStep]"
        @click="nextStep"
      >
        Next
      </Button>
    </div>
  </div>
</template>
```

## 🔌 API Integration

### Endpoint: POST /api/v1/topology

**Request:**
```json
{
  "equipmentId": "EXC-001",
  "equipmentName": "Komatsu PC200-8",
  "equipmentType": "excavator",
  "operatingHours": 5000,
  "components": [
    {
      "componentId": "pump_main_1",
      "componentType": "piston_pump",
      "sensors": ["pressure_in", "pressure_out", "temperature"],
      "nominalPressureBar": 280,
      "nominalFlowLpm": 120,
      "metadata": {
        "manufacturer": "Bosch Rexroth",
        "model": "A10VSO"
      }
    },
    {
      "componentId": "valve_main",
      "componentType": "directional_valve",
      "sensors": ["position"]
    }
  ],
  "edges": [
    {
      "sourceId": "pump_main_1",
      "targetId": "valve_main",
      "edgeType": "high_pressure_hose",
      "diameterMm": 16.0,
      "lengthM": 2.5,
      "material": "steel",
      "flowDirection": "unidirectional"
    }
  ],
  "topologyVersion": "v1.0"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "topologyId": "topo-123abc",
  "equipmentId": "EXC-001",
  "componentsCount": 2,
  "edgesCount": 1,
  "message": "Topology created successfully"
}
```

**Response (400 Bad Request):**
```json
{
  "status": "error",
  "errorCode": "VALIDATION_ERROR",
  "errors": [
    {
      "field": "components[0].nominalPressureBar",
      "message": "Pressure must be ≤1000 bar"
    }
  ]
}
```

## 📝 TODO

- [ ] **Step 1**: SchemaUpload.vue (CSV/JSON/Excel import)
- [ ] Graph visualization (D3.js/Cytoscape.js) на Step 3
- [ ] Версионирование топологий (v1.0, v1.1, v2.0)
- [ ] Шаблоны (Komatsu, CAT, Volvo)
- [ ] Экспорт топологии в JSON/CSV
- [ ] Unit tests для каждого step
- [ ] E2E тест полного wizard flow

## 🎨 UI Components Used

- `Card.vue` - контейнеры для форм
- `Input.vue` - текстовые поля
- `Select.vue` - dropdowns (ComponentType, EdgeType, Material)
- `Checkbox.vue` - quick disconnect, опции
- `Button.vue` - навигация, actions
- `Badge.vue` - счётчики, статусы
- `Table.vue` - списки components/edges
- `Modal.vue` - Add/Edit forms
- `ProgressIndicator.vue` - wizard progress

## 📚 References

- [GNN Service Spec](../../docs/gnn-service-spec.md)
- [Frontend Tactical Guide](../../docs/frontend-tactical-guide.md)
- [Wizard Integration Spec](../../docs/e17fb487.md)
