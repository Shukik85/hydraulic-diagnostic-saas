/**
 * Systems Domain Types
 * @module types/systems
 * @description Enterprise-grade TypeScript types for systems management
 * Strict WCAG 2.1 Level AA accessibility compliance
 */

/**
 * System operational status
 */
export type SystemStatus = 'online' | 'degraded' | 'offline'

/**
 * Sensor type enumeration
 * Corresponds to hydraulic diagnostic sensor types
 */
export enum SensorType {
  PRESSURE = 'pressure',
  TEMPERATURE = 'temperature',
  VIBRATION = 'vibration',
  RPM = 'rpm',
  POSITION = 'position',
  FLOW_RATE = 'flow_rate',
}

/**
 * Sensor status enumeration
 */
export enum SensorStatus {
  OK = 'ok',
  WARNING = 'warning',
  ERROR = 'error',
  OFFLINE = 'offline',
}

/**
 * System list item summary
 * Used for pagination and table display
 */
export interface SystemSummary {
  systemId: string
  equipmentId: string
  equipmentName: string
  equipmentType: string
  status: SystemStatus
  lastUpdateAt: string
  componentsCount: number
  sensorsCount: number
  topologyVersion: string
}

/**
 * Complete system details
 * Includes all metadata and relationships
 */
export interface SystemDetail extends SystemSummary {
  operatingHours: number
  components: SystemComponent[]
  edges: SystemEdge[]
  createdAt: string
  updatedAt: string
  description?: string
  manufacturer?: string
  serialNumber?: string
}

/**
 * System component information
 * Maps to GNN topology nodes
 */
export interface SystemComponent {
  componentId: string
  componentType: string
  name: string
  location: string
  status: SystemStatus
  installationDate?: string
}

/**
 * System edge (connection) information
 * Maps to GNN topology edges
 */
export interface SystemEdge {
  edgeId: string
  sourceComponentId: string
  targetComponentId: string
  edgeType: string
  material?: string
}

/**
 * Sensor real-time reading
 * Includes current state and metadata
 */
export interface SystemSensor {
  sensorId: string
  componentId: string
  sensorType: SensorType
  lastValue: number | string
  unit: string
  status: SensorStatus
  lastUpdateAt: string
  minValue?: number
  maxValue?: number
  normalRange?: [number, number]
  isWarning?: boolean
  isError?: boolean
}

/**
 * Batch sensor readings
 * Used for real-time updates
 */
export interface SensorReadingBatch {
  systemId: string
  readings: SystemSensor[]
  timestamp: string
  batchId: string
}

/**
 * System creation input
 * Request payload for POST /api/v1/systems
 */
export interface SystemCreateInput {
  equipmentId: string
  equipmentName: string
  equipmentType: string
  description?: string
  manufacturer?: string
  serialNumber?: string
  topology: {
    components: SystemComponent[]
    edges: SystemEdge[]
  }
}

/**
 * System update input
 * Request payload for PATCH /api/v1/systems/:id
 */
export interface SystemUpdateInput {
  equipmentName?: string
  description?: string
  operatingHours?: number
}

/**
 * Paginated systems response
 * Standard pagination envelope
 */
export interface PaginatedSystemsResponse {
  status: 'success'
  data: SystemSummary[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

/**
 * Single system response
 */
export interface SystemResponse {
  status: 'success'
  data: SystemDetail
}

/**
 * System creation response
 */
export interface SystemCreateResponse {
  status: 'success'
  data: SystemDetail
  topologyId: string
}

/**
 * Sensor reading response
 */
export interface SensorReadingResponse {
  status: 'success'
  data: SystemSensor[]
  lastUpdate: string
}

/**
 * Error response for API operations
 */
export interface ErrorResponse {
  status: 'error'
  code: string
  message: string
  details?: Record<string, unknown>
}

/**
 * System filter options
 * Used for list queries
 */
export interface SystemFilterOptions {
  search?: string
  status?: SystemStatus[]
  equipmentType?: string[]
  sortBy?: 'name' | 'created' | 'updated'
  sortOrder?: 'asc' | 'desc'
  page?: number
  pageSize?: number
}

/**
 * Sensor filter options
 * Used for sensor list queries
 */
export interface SensorFilterOptions {
  sensorType?: SensorType[]
  status?: SensorStatus[]
  location?: string[]
  component?: string
  sortBy?: 'name' | 'updated'
  sortOrder?: 'asc' | 'desc'
}

/**
 * System metadata for UI display
 * Computed from SystemDetail
 */
export interface SystemMetadata {
  systemId: string
  displayName: string
  statusIcon: string
  statusLabel: string
  statusColor: string
  lastUpdateRelative: string
  healthPercentage: number
}
