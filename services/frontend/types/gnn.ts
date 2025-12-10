/**
 * TypeScript Types для GNN Wizard и Graph Topology
 *
 * Айти должны соответствовать API контракту
 * готовящемся GNN Service (services/gnn_service/schemas.py)
 *
 * @see https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/GNN-API
 */

// ===================== ENUMS =====================

export type ComponentType =
  | 'pump'
  | 'motor'
  | 'cylinder'
  | 'valve'
  | 'filter'
  | 'tank'
  | 'manifold'
  | 'cooler'
  | 'accumulator'
  | 'custom'

export type EdgeType =
  | 'pressure_line'
  | 'return_line'
  | 'drain_line'
  | 'pilot_line'
  | 'measurement_line'

export type MediumType =
  | 'mineral_oil'
  | 'synthetic_oil'
  | 'water_glycol'
  | 'other'

export type SensorType =
  | 'pressure'
  | 'temperature'
  | 'flow_rate'
  | 'vibration'
  | 'position'
  | 'rpm'

export type SensorLocationTarget = 'component' | 'edge'

// ===================== DOMAIN MODELS =====================

/**
 * Компонент графа (узел, оборудование)
 */
export interface TopologyComponent {
  id: string // Локальный id внутри топологии
  name: string // Human-readable name
  type: ComponentType
  description?: string

  // ОборудованиеНые идентификаторы
  equipment_tag?: string
  subsystem?: string

  // Опциональная визуализация
  position?: {
    x: number
    y: number
  }

  // Технические параметры
  specs?: {
    rated_pressure_bar?: number
    nominal_flow_l_min?: number
    max_temperature_c?: number
    volume_liters?: number
    manufacturer?: string
    model?: string
  }

  // Пользовательские атрибуты
  attributes?: Record<string, any>
}

/**
 * Ребро (связь, трубопровод)
 */
export interface TopologyEdge {
  id: string
  source_component_id: string // Муст существовать в components
  target_component_id: string

  type: EdgeType
  medium?: MediumType
  bidirectional?: boolean // дефаольт false

  // Физические параметры
  length_m?: number
  inner_diameter_mm?: number
  roughness_mm?: number

  // Логические показатели
  can_isolate?: boolean
  safety_critical?: boolean

  attributes?: Record<string, any>
}

/**
 * Физический датчик в графе
 */
export interface TopologySensor {
  id: string
  tag: string // Уникальный инженерный тег
  type: SensorType
  unit: string // бар, °C, l/min, ...

  // Привязка к элементу графа
  target_type: SensorLocationTarget
  target_id: string // id компонента или ребра

  // Пороговые значения
  warning_min?: number
  warning_max?: number
  alarm_min?: number
  alarm_max?: number

  description?: string
  installed_at?: string // ISO8601
  manufacturer?: string
  model?: string

  attributes?: Record<string, any>
}

/**
 * Основная модель графа
 * итого продукт GNN Wizard
 */
export interface GraphTopology {
  // ID генерируется сервером (UUID)
  id?: string

  // Обязательные поля
  system_id: number // от HydraulicSystem
  name: string
  description?: string

  // Опциональная версия
  version?: string // аотогенерируется как v1.0

  // Основное содержимое
  components: TopologyComponent[]
  edges: TopologyEdge[]
  sensors: TopologySensor[]

  // Metadata и auditing
  metadata?: Record<string, any>
  created_at?: string // ISO8601, генерируется сервером
  updated_at?: string // ISO8601
}

// ===================== REQUEST/RESPONSE TYPES =====================

/**
 * POST /api/v1/topology - сохранить топологию
 */
export interface CreateTopologyRequest extends Omit<GraphTopology, 'id' | 'created_at' | 'updated_at'> {}

export interface CreateTopologyResponse extends GraphTopology {}

/**
 * POST /api/v1/topology/validate - проверить топологию (драй-ран)
 */
export interface ValidationError {
  code: string // INVALID_REFERENCE, DUPLICATE_COMPONENT_ID, etc.
  message: string
  field?: string
}

export interface ValidationWarning {
  code: string // NO_SENSORS_ON_CRITICAL_EDGE, etc.
  message: string
  field?: string
}

export interface ValidateTopologyRequest extends Omit<GraphTopology, 'id' | 'created_at' | 'updated_at'> {}

export interface ValidateTopologyResponse {
  valid: boolean
  errors: ValidationError[]
  warnings: ValidationWarning[]
}

/**
 * GET /api/v1/systems/{system_id}/topologies - лист топологий
 */
export interface TopologyListItem {
  id: string
  system_id: number
  version?: string
  name: string
  description?: string
  created_at: string
  updated_at: string
}

export interface TopologyListResponse {
  results: TopologyListItem[]
  total: number
  page?: number
  page_size?: number
}

/**
 * GET /api/v1/topology/{topology_id} - детали топологии
 */
export interface GetTopologyResponse extends GraphTopology {}

// ===================== WIZARD STATE =====================

/**
 * Состояние Wizard'a при строительстве
 */
export interface GNNWizardState {
  currentStep: number // 0-4
  isComplete: boolean
  topology: Partial<GraphTopology> // Draft
  errors: Record<string, string[]> // Field-level errors
  savedTopologyId?: string // After successful save
}

/**
 * CSV нагруженные данные до процессинга
 */
export interface CSVImportData {
  components: Array<{
    id: string
    name: string
    type: string
    specs?: Record<string, any>
  }>
  edges: Array<{
    id: string
    source: string
    target: string
    type: string
  }>
  sensors?: Array<{
    id: string
    tag: string
    type: string
    target_id: string
  }>
}

// ===================== HELPERS & CONSTANTS =====================

export const COMPONENT_TYPE_LABELS: Record<ComponentType, string> = {
  pump: 'Носос',
  motor: 'Мотор',
  cylinder: 'Цилиндр',
  valve: 'Клапан',
  filter: 'Фильтр',
  tank: 'Бак',
  manifold: 'Многоячея распределительная',
  cooler: 'Охладитель',
  accumulator: 'Аккумулятор',
  custom: 'Пользовательское',
}

export const SENSOR_TYPE_LABELS: Record<SensorType, string> = {
  pressure: 'Датчик давления',
  temperature: 'датчик температуры',
  flow_rate: 'датчик расхода',
  vibration: 'датчик вибрации',
  position: 'датчик позиции',
  rpm: 'датчик оборотов',
}

export const WIZARD_STEPS = [
  { number: 0, title: 'Equipment Information', subtitle: 'Basic system details' },
  { number: 1, title: 'Schema Upload', subtitle: 'Import components & edges (CSV/JSON)' },
  { number: 2, title: 'Components Editor', subtitle: 'Edit and refine equipment' },
  { number: 3, title: 'Topology Editor', subtitle: 'Define connections & edges' },
  { number: 4, title: 'Review & Submit', subtitle: 'Validate and save' },
] as const
