// types/metadata.ts - TypeScript типы для метаданных гидросистемы

export type ComponentType =
    | 'pump'
    | 'motor'
    | 'cylinder'
    | 'valve'
    | 'filter'
    | 'accumulator';

export type EquipmentType =
    | 'excavator_tracked'
    | 'excavator_wheeled'
    | 'loader_wheeled'
    | 'crane_mobile'
    | 'other';

export interface NormalRange {
    min: number;
    max: number;
    unit: string;
}

export interface ComponentMetadata {
    id: string;
    component_type: ComponentType;
    manufacturer?: string;
    model?: string;

    // Физические характеристики
    nominal_capacity?: number;
    max_pressure?: number;
    max_temperature?: number;

    // Нормальные диапазоны
    normal_ranges: {
        pressure?: NormalRange;
        temperature?: NormalRange;
        flow_rate?: NormalRange;
        vibration?: NormalRange;
    };

    // Топология
    connected_to: string[];
    connection_types: Record<string, 'pressure_line' | 'return_line' | 'pilot_line'>;
    position?: { x: number; y: number };

    // История обслуживания
    install_date?: string;
    last_maintenance?: string;
    maintenance_interval_hours?: number;
    operating_hours?: number;

    // Специфичные для типа
    pump_specific?: PumpSpecific;
    motor_specific?: MotorSpecific;
    cylinder_specific?: CylinderSpecific;
    valve_specific?: ValveSpecific;
    filter_specific?: FilterSpecific;
    accumulator_specific?: AccumulatorSpecific;

    // Confidence scores
    confidence_scores: Record<string, number>;
}

export interface PumpSpecific {
    pump_type: 'axial_piston' | 'gear' | 'vane' | 'radial_piston';
    nominal_flow_rate?: number; // л/мин
    volumetric_efficiency?: number;
    mechanical_efficiency?: number;
    regulation_type?: 'pressure_compensator' | 'load_sensing' | 'fixed';
    max_swash_angle?: number;
}

export interface MotorSpecific {
    motor_type: 'axial_piston' | 'radial_piston' | 'vane' | 'gear';
    displacement: number; // см³/об
    load_character: 'constant' | 'variable' | 'impact' | 'cyclic';
    spike_frequency?: number; // раз/мин (для variable/impact)
    spike_amplitude?: number; // % от P_max
}

export interface CylinderSpecific {
    function: 'primary' | 'auxiliary' | 'balancing' | 'cable';
    piston_diameter: number; // мм
    stroke_length: number; // мм
    rod_area?: number; // мм²
    failure_mode: 'internal_leak' | 'external_leak' | 'seizure' | 'rupture';
    movement_character: 'smooth' | 'fast_switching' | 'pulsed';
}

export interface ValveSpecific {
    valve_type: 'relief' | 'directional' | 'check' | 'throttle' | 'reducing';
    nominal_flow_rate: number; // л/мин
    pressure_setpoint?: number; // бар (для relief/reducing)
    state: 'operational' | 'needs_adjustment' | 'under_replacement' | 'unknown';
}

export interface FilterSpecific {
    location: 'suction' | 'pressure' | 'return' | 'pilot';
    filtration_rating: number; // микроны
    flow_capacity: number; // л/мин
    last_replacement?: string;
    replacement_interval_hours?: number;
}

export interface AccumulatorSpecific {
    accumulator_type: 'hydropneumatic' | 'piston' | 'diaphragm' | 'spring';
    volume: number; // литры
    precharge_pressure: number; // бар
    function: 'pulsation_dampening' | 'emergency_power' | 'energy_recovery' | 'shock_absorption';
    nitrogen_check_date?: string;
}

export interface SystemMetadata {
    // Базовая идентификация
    equipment_id: string;
    equipment_type: EquipmentType;
    manufacturer: string;
    model: string;
    serial_number: string;
    manufacture_date: string;

    // Архитектура системы
    components: ComponentMetadata[];
    adjacency_matrix: number[][];

    // Профиль нагрузки
    duty_cycle?: DutyCycle;

    // Диагностика
    observed_problems: string[];
    sensor_config?: SensorConfig;

    // Метаинформация
    completeness: number; // 0-100%
    last_updated: string;
}

export interface DutyCycle {
    profile_type: 'earthmoving' | 'loading' | 'lifting' | 'custom';
    load_distribution: Record<string, number>; // { 'digging': 40, 'swing': 30, ... }
    peak_load_frequency: 'rare' | 'regular' | 'frequent' | 'constant';
    break_interval_minutes: number;
    ambient_conditions: {
        temp_min: number;
        temp_max: number;
        dusty: boolean;
        humid: boolean;
        high_vibration: boolean;
        hot_environment: boolean;
        high_altitude: boolean;
    };
}

export interface SensorConfig {
    pressure_sensors: SensorDetails[];
    temperature_sensors: SensorDetails[];
    flow_sensors: SensorDetails[];
    vibration_sensors: SensorDetails[];
}

export interface SensorDetails {
    component_id: string;
    sensor_type: 'analog_4_20ma' | 'digital_can' | 'digital_rs485' | 'other';
    measurement_range: { min: number; max: number; unit: string };
    accuracy_percent: number;
    sampling_rate_hz: number;
}

export interface IncompleteDataReport {
    critical_missing: string[];
    secondary_missing: string[];
    inferred_values: Record<string, {
        value: any;
        method: string;
        confidence: number;
    }>;
}

// Wizard state
export interface WizardState {
    current_level: number; // 1-5
    completed_levels: number[];
    system: Partial<SystemMetadata>;
    incompleteness_report: IncompleteDataReport;
}

// UI helpers
export interface ValidationResult {
    valid: boolean;
    error?: string;
    suggestion?: string;
    confidence?: number;
}

export interface SmartDefault {
    value: any;
    reasoning: string;
    confidence: number;
}
