<template>
  <ClientOnly>
    <div v-if="mounted" class="digital-twin">
      <div ref="container" class="canvas-container">
        <canvas ref="canvas"></canvas>
        <div v-if="isLoading" class="loading-overlay">
          <div class="spinner"></div>
          <p>Loading Digital Twin...</p>
        </div>
      </div>
      
      <div class="control-panel">
        <h2>🎮 Controls</h2>
        
        <div class="control-group">
          <label><span class="label-icon">⬆️</span> Boom</label>
          <input type="range" v-model.number="boomTarget" min="0" max="100" @input="onBoomMove" class="control-slider" />
          <div class="value-display">{{ equipment.cylinder_boom.position.toFixed(1) }}%</div>
        </div>
        
        <div class="control-group">
          <label><span class="label-icon">↘️</span> Stick</label>
          <input type="range" v-model.number="stickTarget" min="0" max="100" @input="onStickMove" class="control-slider" />
          <div class="value-display">{{ equipment.cylinder_stick.position.toFixed(1) }}%</div>
        </div>
        
        <div class="control-group">
          <label><span class="label-icon">🪣</span> Bucket</label>
          <input type="range" v-model.number="bucketTarget" min="0" max="100" @input="onBucketMove" class="control-slider" />
          <div class="value-display">{{ equipment.cylinder_bucket.position.toFixed(1) }}°</div>
        </div>

        <div class="quick-actions">
          <button @click="resetPosition" class="btn-secondary">🔄 Reset</button>
          <button @click="simulateFault" class="btn-danger">⚠️ Fault</button>
        </div>
      </div>
      
      <div class="sensor-dashboard">
        <h2>📊 Live Sensors</h2>
        <div class="sensor-grid">
          <div class="sensor-card">
            <div class="sensor-header">
              <div class="sensor-icon">💧</div>
              <h3>Pump</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row">
                <span class="label">RPM:</span>
                <span class="value">{{ equipment.pump.speed_rpm.toFixed(0) }}</span>
              </div>
              <div class="sensor-row">
                <span class="label">Pressure:</span>
                <span class="value">{{ equipment.pump.pressure_outlet.toFixed(1) }} bar</span>
              </div>
            </div>
          </div>
          
          <div class="sensor-card" :class="{ fault: equipment.cylinder_boom.fault }">
            <div class="sensor-header">
              <div class="sensor-icon">🔧</div>
              <h3>Boom Cyl</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row">
                <span class="label">Pos:</span>
                <span class="value">{{ equipment.cylinder_boom.position.toFixed(1) }}%</span>
              </div>
              <div class="sensor-row">
                <span class="label">Press:</span>
                <span class="value" :class="{ warning: equipment.cylinder_boom.pressure > 200 }">
                  {{ equipment.cylinder_boom.pressure.toFixed(1) }} bar
                </span>
              </div>
              <div class="sensor-row">
                <span class="label">Temp:</span>
                <span class="value">{{ equipment.cylinder_boom.temperature.toFixed(1) }}°C</span>
              </div>
            </div>
            <div v-if="equipment.cylinder_boom.fault" class="fault-badge">⚠️ FAULT</div>
          </div>

          <div class="sensor-card">
            <div class="sensor-header">
              <div class="sensor-icon">🔩</div>
              <h3>Stick Cyl</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row">
                <span class="label">Pos:</span>
                <span class="value">{{ equipment.cylinder_stick.position.toFixed(1) }}%</span>
              </div>
              <div class="sensor-row">
                <span class="label">Press:</span>
                <span class="value">{{ equipment.cylinder_stick.pressure.toFixed(1) }} bar</span>
              </div>
            </div>
          </div>

          <div class="sensor-card">
            <div class="sensor-header">
              <div class="sensor-icon">🪣</div>
              <h3>Bucket Cyl</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row">
                <span class="label">Pos:</span>
                <span class="value">{{ equipment.cylinder_bucket.position.toFixed(1) }}°</span>
              </div>
              <div class="sensor-row">
                <span class="label">Press:</span>
                <span class="value">{{ equipment.cylinder_bucket.pressure.toFixed(1) }} bar</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <Transition name="slide-up">
        <div v-if="latestPrediction" class="prediction-panel" :class="latestPrediction.fault_detected ? 'fault' : 'normal'">
          <div class="prediction-content">
            <div class="status-icon">{{ latestPrediction.fault_detected ? '🔴' : '🟢' }}</div>
            <div class="prediction-details">
              <h3>{{ latestPrediction.fault_detected ? 'Fault Detected' : 'Normal' }}</h3>
              <div class="confidence-text">Confidence: {{ (latestPrediction.confidence * 100).toFixed(1) }}%</div>
              <div class="reasoning">{{ latestPrediction.reasoning }}</div>
            </div>
          </div>
        </div>
      </Transition>
    </div>
  </ClientOnly>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { useDigitalTwin } from '~/composables/useDigitalTwin'

const container = ref<HTMLDivElement>()
const canvas = ref<HTMLCanvasElement>()
const mounted = ref(false)
const isLoading = ref(true)

const { equipment, latestPrediction, updatePhysics, moveBoom, moveStick, moveBucket } = useDigitalTwin()

const boomTarget = ref(0)
const stickTarget = ref(0)
const bucketTarget = ref(0)

let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls
let basePart: THREE.Mesh
let boomPart: THREE.Mesh
let stickPart: THREE.Mesh
let bucketPart: THREE.Mesh
let boomCylinder: THREE.Group
let animationId: number
let lastTime = Date.now()

onMounted(async () => {
  mounted.value = true
  await nextTick()
  
  if (!container.value || !canvas.value) {
    console.error('Container not ready')
    return
  }
  
  try {
    initThreeJS()
    createExcavator()
    createCylinders()
    startAnimation()
    setTimeout(() => { isLoading.value = false }, 300)
  } catch (error) {
    console.error('Init failed:', error)
  }
})

function initThreeJS() {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x1a1a2e)
  
  const width = container.value!.clientWidth
  const height = container.value!.clientHeight
  
  camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000)
  camera.position.set(20, 15, 20)
  
  renderer = new THREE.WebGLRenderer({ canvas: canvas.value!, antialias: true })
  renderer.setSize(width, height)
  renderer.shadowMap.enabled = true
  
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
  scene.add(ambientLight)
  
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(30, 40, 20)
  directionalLight.castShadow = true
  scene.add(directionalLight)
  
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(100, 100),
    new THREE.MeshStandardMaterial({ color: 0x2d3436 })
  )
  ground.rotation.x = -Math.PI / 2
  ground.receiveShadow = true
  scene.add(ground)
}

function createExcavator() {
  const yellow = 0xffb302
  const mat = (color: number) => new THREE.MeshStandardMaterial({ 
    color, metalness: 0.4, roughness: 0.6 
  })
  
  basePart = new THREE.Mesh(new THREE.BoxGeometry(4, 1.5, 3), mat(yellow))
  basePart.position.y = 0.75
  basePart.castShadow = true
  scene.add(basePart)
  
  boomPart = new THREE.Mesh(new THREE.BoxGeometry(8, 0.8, 0.8), mat(yellow))
  boomPart.position.set(4, 0, 0)
  boomPart.castShadow = true
  
  const boomPivot = new THREE.Object3D()
  boomPivot.position.set(0, 2, 0)
  boomPivot.add(boomPart)
  basePart.add(boomPivot)
  
  stickPart = new THREE.Mesh(new THREE.BoxGeometry(6, 0.6, 0.6), mat(yellow))
  stickPart.position.set(3, 0, 0)
  stickPart.castShadow = true
  
  const stickPivot = new THREE.Object3D()
  stickPivot.position.set(8, 0, 0)
  stickPivot.add(stickPart)
  boomPart.add(stickPivot)
  
  bucketPart = new THREE.Mesh(new THREE.BoxGeometry(2, 1.8, 2), mat(0x555555))
  bucketPart.position.set(2.5, 0, 0)
  bucketPart.castShadow = true
  
  const bucketPivot = new THREE.Object3D()
  bucketPivot.position.set(6, 0, 0)
  bucketPivot.add(bucketPart)
  stickPart.add(bucketPivot)
  
  ;(boomPart as any).pivot = boomPivot
  ;(stickPart as any).pivot = stickPivot
  ;(bucketPart as any).pivot = bucketPivot
}

function createCylinders() {
  const gray = 0x888888
  
  boomCylinder = new THREE.Group()
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.2, 0.2, 3, 16),
    new THREE.MeshStandardMaterial({ color: gray, metalness: 0.6 })
  )
  body.castShadow = true
  boomCylinder.add(body)
  
  const rod = new THREE.Mesh(
    new THREE.CylinderGeometry(0.15, 0.15, 1.5, 16),
    new THREE.MeshStandardMaterial({ color: 0xaaaaaa, metalness: 0.8 })
  )
  rod.position.y = 2
  rod.castShadow = true
  boomCylinder.add(rod)
  
  ;(boomCylinder as any).rod = rod
  
  boomCylinder.position.set(1, 0.5, 0)
  boomCylinder.rotation.z = Math.PI / 6
  basePart.add(boomCylinder)
}

function startAnimation() {
  function animate() {
    animationId = requestAnimationFrame(animate)
    
    const now = Date.now()
    const delta = Math.min((now - lastTime) / 1000, 0.1)
    lastTime = now
    
    updatePhysics(delta)
    updatePose()
    
    controls.update()
    renderer.render(scene, camera)
  }
  animate()
}

function updatePose() {
  if (!boomPart) return
  
  const boomAngle = (equipment.cylinder_boom.position / 100) * (Math.PI / 3)
  ;(boomPart as any).pivot.rotation.z = boomAngle
  
  if (equipment.cylinder_boom.fault) {
    (boomPart.material as THREE.MeshStandardMaterial).emissive.setHex(0xff0000)
    (boomPart.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.5
  } else {
    (boomPart.material as THREE.MeshStandardMaterial).emissiveIntensity = 0
  }
  
  const stickAngle = -(equipment.cylinder_stick.position / 100) * (Math.PI / 2)
  ;(stickPart as any).pivot.rotation.z = stickAngle
  
  const bucketAngle = (equipment.cylinder_bucket.position / 100) * (Math.PI / 4)
  ;(bucketPart as any).pivot.rotation.z = bucketAngle
  
  if (boomCylinder) {
    const extension = equipment.cylinder_boom.position / 100
    ;(boomCylinder as any).rod.position.y = 1.5 + extension * 1.5
  }
}

async function onBoomMove() { await moveBoom(boomTarget.value) }
async function onStickMove() { await moveStick(stickTarget.value) }
async function onBucketMove() { await moveBucket(bucketTarget.value) }

function resetPosition() {
  boomTarget.value = 0
  stickTarget.value = 0
  bucketTarget.value = 0
  Object.assign(equipment.cylinder_boom, { position: 0, fault: false })
  Object.assign(equipment.cylinder_stick, { position: 0, fault: false })
  Object.assign(equipment.cylinder_bucket, { position: 0, fault: false })
}

function simulateFault() {
  equipment.cylinder_boom.pressure = 250
  equipment.cylinder_boom.fault = true
  latestPrediction.value = {
    fault_detected: true,
    confidence: 0.95,
    fault_type: 'overpressure',
    reasoning: 'Boom pressure exceeded 220 bar'
  }
  setTimeout(() => {
    equipment.cylinder_boom.fault = false
    latestPrediction.value = null
  }, 5000)
}

onBeforeUnmount(() => {
  if (animationId) cancelAnimationFrame(animationId)
  if (renderer) renderer.dispose()
  if (controls) controls.dispose()
})
</script>

<style scoped>
.digital-twin {
  display: grid;
  grid-template-columns: 2fr 1fr;
  grid-template-rows: 1fr auto;
  gap: 20px;
  height: 100vh;
  padding: 20px;
  background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
}

.canvas-container {
  position: relative;
  grid-column: 1;
  grid-row: 1 / 3;
  border-radius: 16px;
  overflow: hidden;
  background: #0f0f1e;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.loading-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(15, 15, 30, 0.95);
  color: white;
  font-size: 18px;
  z-index: 10;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid rgba(255, 255, 255, 0.1);
  border-top-color: #ffb302;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 20px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.control-panel {
  grid-column: 2;
  grid-row: 1;
  background: rgba(26, 26, 46, 0.95);
  border-radius: 16px;
  padding: 24px;
  color: white;
  overflow-y: auto;
}

.control-panel h2 {
  margin: 0 0 24px 0;
  font-size: 20px;
  font-weight: 700;
}

.control-group {
  margin-bottom: 24px;
}

.control-group label {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  font-weight: 600;
  color: #10b981;
  font-size: 14px;
}

.label-icon {
  font-size: 20px;
}

.control-slider {
  width: 100%;
  height: 8px;
  border-radius: 4px;
  background: #2d2d3f;
  outline: none;
  -webkit-appearance: none;
  margin-bottom: 8px;
}

.control-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #ffb302, #ff8c00);
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(255, 179, 2, 0.5);
}

.value-display {
  font-size: 16px;
  font-weight: 600;
  color: #e5e7eb;
}

.quick-actions {
  display: flex;
  gap: 12px;
  margin-top: 24px;
}

.quick-actions button {
  flex: 1;
  padding: 12px;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-secondary {
  background: #374151;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.btn-danger {
  background: #ef4444;
  color: white;
}

.btn-danger:hover {
  background: #dc2626;
}

.sensor-dashboard {
  grid-column: 2;
  grid-row: 2;
  background: rgba(26, 26, 46, 0.95);
  border-radius: 16px;
  padding: 24px;
  color: white;
  max-height: 450px;
  overflow-y: auto;
}

.sensor-dashboard h2 {
  margin: 0 0 16px 0;
  font-size: 18px;
}

.sensor-grid {
  display: grid;
  gap: 12px;
}

.sensor-card {
  background: #1f1f2e;
  border-radius: 8px;
  padding: 12px;
  border: 2px solid transparent;
  transition: all 0.3s;
}

.sensor-card.fault {
  border-color: #ef4444;
  animation: pulse-border 1s infinite;
}

@keyframes pulse-border {
  0%, 100% { border-color: #ef4444; }
  50% { border-color: #dc2626; }
}

.sensor-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.sensor-icon {
  font-size: 24px;
}

.sensor-header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
}

.sensor-values {
  font-size: 13px;
}

.sensor-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}

.sensor-row .label {
  color: #9ca3af;
}

.sensor-row .value {
  font-weight: 600;
  color: #10b981;
}

.sensor-row .value.warning {
  color: #f59e0b;
}

.fault-badge {
  margin-top: 8px;
  padding: 6px;
  background: #ef4444;
  border-radius: 4px;
  font-weight: 700;
  font-size: 11px;
  text-align: center;
  animation: pulse 1s infinite;
}

.prediction-panel {
  position: fixed;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.95);
  border-radius: 16px;
  padding: 20px;
  color: white;
  min-width: 400px;
  backdrop-filter: blur(20px);
  z-index: 100;
}

.prediction-panel.fault {
  border: 2px solid #ef4444;
}

.prediction-panel.normal {
  border: 2px solid #10b981;
}

.prediction-content {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.status-icon {
  font-size: 40px;
}

.prediction-details h3 {
  margin: 0 0 8px 0;
  font-size: 18px;
}

.confidence-text {
  font-size: 14px;
  color: #10b981;
  margin-bottom: 8px;
}

.reasoning {
  font-size: 13px;
  color: #9ca3af;
}

.slide-up-enter-active,
.slide-up-leave-active {
  transition: all 0.3s;
}

.slide-up-enter-from,
.slide-up-leave-to {
  transform: translateX(-50%) translateY(100px);
  opacity: 0;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}
</style>
