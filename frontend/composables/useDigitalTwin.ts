// composables/useDigitalTwin.ts - Physics-based Digital Twin
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

// Физические параметры экскаватора CAT 336
const PHYSICS = {
  boom: { 
    mass: 500,        // kg
    length: 8,        // m
    cog: 4,          // center of gravity от pivot
    leverArm: 2.0    // плечо цилиндра
  },
  stick: { mass: 300, length: 6, cog: 3, leverArm: 1.5 },
  bucket: { mass: 200, length: 2, cog: 1, leverArm: 0.8 },
  cylinderArea: 0.02,  // m² площадь поршня
  gravity: 9.81        // m/s²
}

export function useDigitalTwin() {
  const equipment = reactive<EquipmentState>({
    cylinder_boom: { position: 0, velocity: 0, pressure: 50, temperature: 60, fault: false },
    cylinder_stick: { position: 0, velocity: 0, pressure: 50, temperature: 60, fault: false },
    cylinder_bucket: { position: 0, velocity: 0, pressure: 50, temperature: 60, fault: false },
    pump: { 
      speed_rpm: 1800, 
      pressure_outlet: 180, 
      temperature: 65, 
      power: 45, 
      vibration: 2.1, 
      fault: false 
    },
    motor_swing: { 
      position: 0, 
      velocity: 0, 
      pressure: 50, 
      temperature: 70, 
      fault: false, 
      angle: 0 
    }
  })

  const latestPrediction = ref<any>(null)

  // 🎯 РЕАЛИСТИЧНЫЙ РАСЧЁТ ДАВЛЕНИЯ на основе момента силы
  function calculatePressure(component: 'boom' | 'stick' | 'bucket'): number {
    const boomAngle = (equipment.cylinder_boom.position / 100) * (Math.PI / 3)
    const stickAngle = (equipment.cylinder_stick.position / 100) * (Math.PI / 2)
    const bucketAngle = (equipment.cylinder_bucket.position / 100) * (Math.PI / 4)

    let totalMoment = 0

    // Момент от boom
    const boomHorizontal = Math.cos(boomAngle)
    totalMoment += PHYSICS.boom.mass * PHYSICS.gravity * PHYSICS.boom.cog * boomHorizontal

    // Момент от stick (зависит от положения boom)
    const stickX = PHYSICS.boom.length * Math.cos(boomAngle) +
                   PHYSICS.stick.length * Math.cos(boomAngle - stickAngle)
    totalMoment += PHYSICS.stick.mass * PHYSICS.gravity * stickX

    // Момент от bucket
    const bucketX = stickX + PHYSICS.bucket.cog * Math.cos(boomAngle - stickAngle - bucketAngle)
    totalMoment += PHYSICS.bucket.mass * PHYSICS.gravity * bucketX

    // Сила в цилиндре
    const leverArm = PHYSICS[component].leverArm
    const force = totalMoment / leverArm

    // Давление
    const pressurePa = force / PHYSICS.cylinderArea
    const pressureBar = pressurePa / 100000

    return Math.max(50, Math.min(280, 50 + pressureBar))
  }

  function updatePhysics(deltaTime: number) {
    // Update cylinders
    updateCylinder(equipment.cylinder_boom, deltaTime, 'boom')
    updateCylinder(equipment.cylinder_stick, deltaTime, 'stick')
    updateCylinder(equipment.cylinder_bucket, deltaTime, 'bucket')

    // Pump adapts to demand
    const demand = [
      equipment.cylinder_boom,
      equipment.cylinder_stick,
      equipment.cylinder_bucket
    ].reduce((sum, c) => sum + Math.abs(c.velocity), 0)

    equipment.pump.speed_rpm = 1800 + demand * 200
    equipment.pump.pressure_outlet = 180 + demand * 20

    if (demand > 0.1) {
      equipment.pump.temperature = Math.min(95, equipment.pump.temperature + deltaTime * 2)
    } else {
      equipment.pump.temperature = Math.max(65, equipment.pump.temperature - deltaTime * 0.5)
    }
  }

  function updateCylinder(
    cylinder: ComponentState, 
    deltaTime: number, 
    type: 'boom' | 'stick' | 'bucket'
  ) {
    if (Math.abs(cylinder.velocity) > 0.01) {
      // Movement
      cylinder.position += cylinder.velocity * deltaTime * 50
      cylinder.position = Math.max(0, Math.min(100, cylinder.position))

      // Temperature rises
      cylinder.temperature = Math.min(90, cylinder.temperature + deltaTime)

      // Deceleration
      cylinder.velocity *= 0.95
    } else {
      cylinder.velocity = 0
      cylinder.temperature = Math.max(60, cylinder.temperature - deltaTime * 0.5)
    }

    // Реалистичное давление (всегда, даже в покое!)
    cylinder.pressure = calculatePressure(type)
  }

  function moveBoom(target: number) {
    const dist = target - equipment.cylinder_boom.position
    equipment.cylinder_boom.velocity = Math.sign(dist) * Math.min(2.0, Math.abs(dist) / 20)
    return predictFault('cylinder_boom')
  }

  function moveStick(target: number) {
    const dist = target - equipment.cylinder_stick.position
    equipment.cylinder_stick.velocity = Math.sign(dist) * Math.min(2.0, Math.abs(dist) / 20)
    return predictFault('cylinder_stick')
  }

  function moveBucket(target: number) {
    const dist = target - equipment.cylinder_bucket.position
    equipment.cylinder_bucket.velocity = Math.sign(dist) * Math.min(2.0, Math.abs(dist) / 20)
    return predictFault('cylinder_bucket')
  }

  function rotateSwing(angle: number) {
    equipment.motor_swing.angle = angle
    equipment.motor_swing.velocity = (angle - equipment.motor_swing.position) / 100
    return predictFault('motor_swing')
  }

  async function predictFault(component: keyof EquipmentState) {
    const state = equipment[component] as any
    const isFault = state.pressure > 220 || state.temperature > 85

    const prediction = {
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
