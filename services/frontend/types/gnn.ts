/**
 * GNN Service Types
 * Based on Wizard GNN Integration Spec
 */

// Component Types
export enum ComponentType {
  // Pumps
  HYDRAULIC_PUMP = 'hydraulic_pump',
  GEAR_PUMP = 'gear_pump',
  PISTON_PUMP = 'piston_pump',
  VANE_PUMP = 'vane_pump',

  // Valves
  HYDRAULIC_VALVE = 'hydraulic_valve',
  DIRECTIONAL_VALVE = 'directional_valve',
  PRESSURE_RELIEF_VALVE = 'pressure_relief_valve',
  FLOW_CONTROL_VALVE = 'flow_control_valve',
  CHECK_VALVE = 'check_valve',

  // Actuators
  HYDRAULIC_CYLINDER = 'hydraulic_cylinder',
  HYDRAULIC_MOTOR = 'hydraulic_motor',
  ROTARY_ACTUATOR = 'rotary_actuator',

  // Other
  ACCUMULATOR = 'accumulator',
  FILTER = 'filter',
  RESERVOIR = 'reservoir',
  HEAT_EXCHANGER = 'heat_exchanger',
  MANIFOLD = 'manifold',
}

// Edge Types (connections)
export enum EdgeType {
  HYDRAULIC_LINE = 'hydraulic_line',
  HIGH_PRESSURE_HOSE = 'high_pressure_hose',
  LOW_PRESSURE_RETURN = 'low_pressure_return',
  PILOT_LINE = 'pilot_line',
  DRAIN_LINE = 'drain_line',
  MANIFOLD_CONNECTION = 'manifold_connection',
}

// Edge Material
export enum EdgeMaterial {
  STEEL = 'steel',
  RUBBER = 'rubber',
  COMPOSITE = 'composite',
  THERMOPLASTIC = 'thermoplastic',
}

// Flow Direction
export type FlowDirection = 'unidirectional' | 'bidirectional';

// Component Interface
export interface Component {
  componentId: string; // Unique ID (1-50 chars, alphanumeric)
  componentType: ComponentType;
  sensors: string[]; // e.g., ["pressure_in", "pressure_out", "temperature"]
  nominalPressureBar?: number; // 0-1000
  nominalFlowLpm?: number; // 0-1000
  ratedPowerKw?: number; // >= 0
  metadata?: ComponentMetadata;
}

export interface ComponentMetadata {
  manufacturer?: string;
  model?: string;
  serialNumber?: string;
  [key: string]: any;
}

// Edge Interface (connection)
export interface Edge {
  sourceId: string; // Must exist in components
  targetId: string; // Must exist in components, cannot be same as sourceId
  edgeType: EdgeType;
  diameterMm?: number; // 0-500
  lengthM?: number; // 0-1000
  pressureRatingBar?: number; // 0-1000
  material?: EdgeMaterial;
  flowDirection: FlowDirection;
  hasQuickDisconnect?: boolean;
}

// Equipment Metadata
export interface EquipmentMetadata {
  equipmentId: string; // 1-100 chars, alphanumeric
  equipmentName: string; // 1-255 chars
  equipmentType: string; // e.g., "excavator", "loader"
  operatingHours?: number; // >= 0
}

// Graph Topology (full submission)
export interface GraphTopology {
  equipmentId: string;
  equipmentName: string;
  equipmentType?: string;
  operatingHours?: number;
  components: Component[];
  edges: Edge[];
  topologyVersion?: string; // default: "v1.0"
}

// API Response
export interface TopologySubmitResponse {
  status: 'success' | 'error';
  topologyId?: string;
  equipmentId?: string;
  componentsCount?: number;
  edgesCount?: number;
  message?: string;
  errorCode?: string;
  errors?: ValidationError[];
}

export interface ValidationError {
  field: string;
  message: string;
}

// Wizard State
export interface WizardState {
  currentStep: number;
  equipment: EquipmentMetadata;
  components: Component[];
  edges: Edge[];
  isValid: boolean;
}
