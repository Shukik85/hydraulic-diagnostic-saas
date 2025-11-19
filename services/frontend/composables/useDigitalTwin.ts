// composables/useDigitalTwin.ts - Physics-based Digital Twin with Thermodynamics
import { reactive, ref } from 'vue'

export interface ComponentState {
  position: number      // 0-100%
  velocity: number      // m/s
  pressure: number      // bar
  temperature: number   // °C
  fault: boolean
}

export interface EquipmentState {
  cylinder_boom: ComponentState
  cylinder_stick: ComponentState
  cylinder_bucket: ComponentState
  pump: {
    speed_rpm: number
    pressure_outlet: number
    temperature: number
    power: number
    vibration: number
    fault: boolean
  }
  motor_swing: ComponentState & { angle: number }
}

export interface FaultPrediction {
  fault_detected: boolean
  confidence: number
  fault_type: string | null
  reasoning: string
}

export interface UseDigitalTwinReturn {
  equipment: EquipmentState
  latestPrediction: Ref<FaultPrediction | null>
  updatePhysics: (deltaTime: number) => void
  moveBoom: (target: number) => FaultPrediction
  moveStick: (target: number) => FaultPrediction
  moveBucket: (target: number) => FaultPrediction
  rotateSwing: (angle: number) => FaultPrediction
}

// Физические параметры системы
const PHYSICS = {
  boom: { mass: 500, length: 6, cog: 3, leverArm: 1.8 },
  stick: { mass: 300, length: 4.5, cog: 2.25, leverArm: 1.2 },
  bucket: { mass: 200, length: 1.5, cog: 0.75, leverArm: 0.6 },
  cylinderArea: 0.02,  // m²
  gravity: 9.81,
  
  // Термодинамика (трубопровод 1" = 25.4mm, без охладителя)
  thermal: {
    pipeInnerDiameter: 0.0254,     // m (1 inch)
    pipeThermalResistance: 0.15,   // K·m/W
    ambientTemp: 25,               // °C
    oilSpecificHeat: 1900,         // J/(kg·K)
    oilDensity: 870,               // kg/m³
    convectionCoeff: 10            // W/(m²·K) - без принудительного охлаждения
  }
}

/**
 * Physics-based Digital Twin composable for hydraulic systems
 * Provides realistic simulation of hydraulic equipment with thermodynamic calculations
 * 
 * @returns Digital twin instance with equipment state and control methods
 */
export function useDigitalTwin(): UseDigitalTwinReturn {
  const equipment = reactive<EquipmentState>({
    cylinder_boom: { position: 0, velocity: 0, pressure: 50, temperature: 45, fault: false },
    cylinder_stick: { position: 0, velocity: 0, pressure: 50, temperature: 45, fault: false },
    cylinder_bucket: { position: 0, velocity: 0, pressure: 50, temperature: 45, fault: false },
    pump: { speed_rpm: 1800, pressure_outlet: 180, temperature: 50, power: 45, vibration: 2.1, fault: false },
    motor_swing: { position: 0, velocity: 0, pressure: 50, temperature: 50, fault: false, angle: 0 }
  })

  const latestPrediction = ref<FaultPrediction | null>(null)

  // 🌡️ РЕАЛИСТИЧНАЯ ТЕРМОДИНАМИКА
  /**
   * Calculate heat transfer for a component
   * @param component - Component state to update
   * @param flowRate - Flow rate in L/min
   * @param workPower - Work power in kW
   * @returns Temperature change in °C/s
   */
  function calculateHeatTransfer(
    component: ComponentState,
    flowRate: number,  // L/min
    workPower: number  // kW
  ): number {
    const { thermal } = PHYSICS
    
    // Тепловыделение от работы (гидравлические потери ~15%)
    const heatFromWork = workPower * 0.15 * 1000  // Watts
    
    // Объёмный расход в m³/s
    const volumeFlow = (flowRate / 1000) / 60
    
    // Масса масла в системе (примерно)
    const oilMass = volumeFlow * thermal.oilDensity * 10  // кг
    
    // Конвективное охлаждение через стенки труб
    const pipeArea = Math.PI * thermal.pipeInnerDiameter * 5  // 5м трубопровода
    const tempDiff = component.temperature - thermal.ambientTemp
    const heatLoss = thermal.convectionCoeff * pipeArea * tempDiff
    
    // Чистое тепловыделение
    const netHeat = heatFromWork - heatLoss
    
    // Изменение температуры (dT = Q / (m * c))
    const deltaTemp = netHeat / (oilMass * thermal.oilSpecificHeat)
    
    return deltaTemp
  }

  // 🎯 Расчёт давления на основе момента силы
  /**
   * Calculate pressure for a cylinder based on boom position
   * @param component - Component type ('boom' | 'stick' | 'bucket')
   * @returns Pressure in bar
   */
  function calculatePressure(component: 'boom' | 'stick' | 'bucket'): number {
    const boomAngle = (equipment.cylinder_boom.position / 100) * (Math.PI / 3)
    const stickAngle = (equipment.cylinder_stick.position / 100) * (Math.PI / 2.5)
    const bucketAngle = (equipment.cylinder_bucket.position / 100) * (Math.PI / 4)

    let totalMoment = 0

    const boomHorizontal = Math.cos(boomAngle)
    totalMoment += PHYSICS.boom.mass * PHYSICS.gravity * PHYSICS.boom.cog * boomHorizontal

    const stickX = PHYSICS.boom.length * Math.cos(boomAngle) +
                   PHYSICS.stick.length * Math.cos(boomAngle - stickAngle)
    totalMoment += PHYSICS.stick.mass * PHYSICS.gravity * stickX

    const bucketX = stickX + PHYSICS.bucket.cog * Math.cos(boomAngle - stickAngle - bucketAngle)
    totalMoment += PHYSICS.bucket.mass * PHYSICS.gravity * bucketX

    const leverArm = PHYSICS[component].leverArm
    const force = totalMoment / leverArm
    const pressurePa = force / PHYSICS.cylinderArea
    const pressureBar = pressurePa / 100000

    return Math.max(50, Math.min(280, 50 + pressureBar))
  }

  /**
   * Update physics simulation for all equipment
   * @param deltaTime - Time delta in seconds
   */
  function updatePhysics(deltaTime: number): void {
    updateCylinder(equipment.cylinder_boom, deltaTime, 'boom')
    updateCylinder(equipment.cylinder_stick, deltaTime, 'stick')
    updateCylinder(equipment.cylinder_bucket, deltaTime, 'bucket')

    // Общий расход от насоса
    const totalDemand = [
      equipment.cylinder_boom,
      equipment.cylinder_stick,
      equipment.cylinder_bucket
    ].reduce((sum, c) => sum + Math.abs(c.velocity), 0)

    const flowRate = 50 + totalDemand * 100  // L/min
    const pumpPower = (flowRate * equipment.pump.pressure_outlet) / 600  // kW

    equipment.pump.speed_rpm = 1800 + totalDemand * 200
    equipment.pump.pressure_outlet = 180 + totalDemand * 20
    equipment.pump.power = pumpPower

    // Реалистичный нагрев насоса
    const pumpHeatDelta = calculateHeatTransfer(
      { temperature: equipment.pump.temperature } as ComponentState,
      flowRate,
      pumpPower
    )
    
    equipment.pump.temperature = Math.max(
      PHYSICS.thermal.ambientTemp + 10,
      Math.min(95, equipment.pump.temperature + pumpHeatDelta * deltaTime)
    )
  }

  /**
   * Update cylinder state based on velocity and thermal properties
   * @param cylinder - Cylinder component to update
   * @param deltaTime - Time delta in seconds
   * @param type - Cylinder type for pressure calculation
   */
  function updateCylinder(
    cylinder: ComponentState,
    deltaTime: number,
    type: 'boom' | 'stick' | 'bucket'
  ): void {
    if (Math.abs(cylinder.velocity) > 0.01) {
      cylinder.position += cylinder.velocity * deltaTime * 50
      cylinder.position = Math.max(0, Math.min(100, cylinder.position))

      // Нагрев при работе (трение + сжатие масла)
      const workHeat = Math.abs(cylinder.velocity) * cylinder.pressure * 0.002
      cylinder.temperature = Math.min(90, cylinder.temperature + workHeat * deltaTime)

      cylinder.velocity *= 0.93
    } else {
      cylinder.velocity = 0
      
      // Пассивное охлаждение
      const coolRate = (cylinder.temperature - PHYSICS.thermal.ambientTemp) * 0.03
      cylinder.temperature = Math.max(PHYSICS.thermal.ambientTemp + 5, cylinder.temperature - coolRate * deltaTime)
    }

    cylinder.pressure = calculatePressure(type)
  }

  /**
   * Move boom to target position
   * @param target - Target position (0-100)
   * @returns Fault prediction
   */
  function moveBoom(target: number): FaultPrediction {
    const dist = target - equipment.cylinder_boom.position
    equipment.cylinder_boom.velocity = Math.sign(dist) * Math.min(2.5, Math.abs(dist) / 15)
    return predictFault('cylinder_boom')
  }

  /**
   * Move stick to target position
   * @param target - Target position (0-100)
   * @returns Fault prediction
   */
  function moveStick(target: number): FaultPrediction {
    const dist = target - equipment.cylinder_stick.position
    equipment.cylinder_stick.velocity = Math.sign(dist) * Math.min(2.5, Math.abs(dist) / 15)
    return predictFault('cylinder_stick')
  }

  /**
   * Move bucket to target position
   * @param target - Target position (0-100)
   * @returns Fault prediction
   */
  function moveBucket(target: number): FaultPrediction {
    const dist = target - equipment.cylinder_bucket.position
    equipment.cylinder_bucket.velocity = Math.sign(dist) * Math.min(2.5, Math.abs(dist) / 15)
    return predictFault('cylinder_bucket')
  }

  /**
   * Rotate swing motor to target angle
   * @param angle - Target angle in degrees
   * @returns Fault prediction
   */
  function rotateSwing(angle: number): FaultPrediction {
    equipment.motor_swing.angle = angle
    equipment.motor_swing.velocity = (angle - equipment.motor_swing.position) / 100
    return predictFault('motor_swing')
  }

  /**
   * Predict faults based on current component state
   * @param component - Component to analyze
   * @returns Fault prediction with confidence and reasoning
   */
  function predictFault(component: keyof EquipmentState): FaultPrediction {
    const state = equipment[component] as any
    const isFault = state.pressure > 220 || state.temperature > 85

    const prediction: FaultPrediction = {
      fault_detected: isFault,
      confidence: Math.random() * 0.3 + 0.7,
      fault_type: isFault ? (state.pressure > 220 ? 'overpressure' : 'overheating') : null,
      reasoning: isFault
        ? `${state.pressure > 220 ? 'Pressure' : 'Temperature'} exceeded: ${state.pressure?.toFixed(1) || state.temperature?.toFixed(1)}`
        : 'All parameters normal'
    }

    latestPrediction.value = prediction
    state.fault = isFault
    return prediction
  }

  return {
    equipment,
    latestPrediction,
    updatePhysics,
    moveBoom,
    moveStick,
    moveBucket,
    rotateSwing
  }
}
