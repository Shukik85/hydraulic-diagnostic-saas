// composables/useDigitalTwin.ts
import { reactive, ref } from 'vue'

export interface ComponentState {
  position: number
  velocity: number
  pressure: number
  temperature: number
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

export function useDigitalTwin() {
  const equipment = reactive<EquipmentState>({
    cylinder_boom: {
      position: 0,
      velocity: 0,
      pressure: 50,
      temperature: 60,
      fault: false
    },
    cylinder_stick: {
      position: 0,
      velocity: 0,
      pressure: 50,
      temperature: 60,
      fault: false
    },
    cylinder_bucket: {
      position: 0,
      velocity: 0,
      pressure: 50,
      temperature: 60,
      fault: false
    },
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

  function updatePhysics(deltaTime: number) {
    updateCylinderPhysics(equipment.cylinder_boom, deltaTime)
    updateCylinderPhysics(equipment.cylinder_stick, deltaTime)
    updateCylinderPhysics(equipment.cylinder_bucket, deltaTime)
    
    const totalDemand = [
      equipment.cylinder_boom,
      equipment.cylinder_stick,
      equipment.cylinder_bucket
    ].reduce((sum, c) => sum + Math.abs(c.velocity), 0)
    
    equipment.pump.speed_rpm = 1800 + (totalDemand * 200)
    equipment.pump.pressure_outlet = 180 + (totalDemand * 20)
    
    if (totalDemand > 0.1) {
      equipment.pump.temperature = Math.min(95, equipment.pump.temperature + deltaTime * 2)
    } else {
      equipment.pump.temperature = Math.max(65, equipment.pump.temperature - deltaTime * 0.5)
    }
  }

  function updateCylinderPhysics(cylinder: ComponentState, deltaTime: number) {
    if (Math.abs(cylinder.velocity) > 0.01) {
      cylinder.position += cylinder.velocity * deltaTime * 50
      cylinder.position = Math.max(0, Math.min(100, cylinder.position))
      cylinder.pressure = 50 + (Math.abs(cylinder.velocity) * 100)
      cylinder.temperature = Math.min(90, cylinder.temperature + deltaTime * 1)
      cylinder.velocity *= 0.95
    } else {
      cylinder.velocity = 0
      cylinder.temperature = Math.max(60, cylinder.temperature - deltaTime * 0.5)
    }
  }

  function moveBoom(targetPosition: number) {
    const distance = targetPosition - equipment.cylinder_boom.position
    equipment.cylinder_boom.velocity = Math.sign(distance) * Math.min(0.8, Math.abs(distance) / 50)
    return predictFault('cylinder_boom')
  }

  function moveStick(targetPosition: number) {
    const distance = targetPosition - equipment.cylinder_stick.position
    equipment.cylinder_stick.velocity = Math.sign(distance) * Math.min(0.8, Math.abs(distance) / 50)
    return predictFault('cylinder_stick')
  }

  function moveBucket(targetPosition: number) {
    const distance = targetPosition - equipment.cylinder_bucket.position
    equipment.cylinder_bucket.velocity = Math.sign(distance) * Math.min(0.8, Math.abs(distance) / 50)
    return predictFault('cylinder_bucket')
  }

  function rotateSwing(targetAngle: number) {
    equipment.motor_swing.angle = targetAngle
    equipment.motor_swing.velocity = (targetAngle - equipment.motor_swing.position) / 100
    return predictFault('motor_swing')
  }

  async function predictFault(component: keyof EquipmentState) {
    const componentState = equipment[component] as any
    
    const sensorData = {
      timestamp: Date.now(),
      component,
      equipment_id: 'CAT336_DEMO',
      ...(component.includes('cylinder') ? {
        pressure_extend: componentState.pressure,
        pressure_retract: componentState.pressure * 0.3,
        position: componentState.position,
        velocity: componentState.velocity,
        pressure_diff: componentState.pressure * 0.7
      } : {
        speed_rpm: equipment.pump.speed_rpm,
        pressure_outlet: equipment.pump.pressure_outlet,
        temperature: componentState.temperature,
        vibration: equipment.pump.vibration,
        power: equipment.pump.power
      })
    }

    const prediction = await mockMLPrediction(sensorData)
    latestPrediction.value = prediction
    componentState.fault = prediction.fault_detected
    return prediction
  }

  async function mockMLPrediction(sensorData: any) {
    await new Promise(resolve => setTimeout(resolve, 100))
    
    const isFault = 
      sensorData.pressure_extend > 200 || 
      sensorData.temperature > 85
    
    return {
      fault_detected: isFault,
      confidence: Math.random() * 0.3 + 0.7,
      fault_type: isFault ? 'high_pressure' : null,
      reasoning: isFault 
        ? `Excessive pressure detected: ${sensorData.pressure_extend?.toFixed(1)} bar (threshold: 200 bar)` 
        : 'All parameters within normal range'
    }
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
