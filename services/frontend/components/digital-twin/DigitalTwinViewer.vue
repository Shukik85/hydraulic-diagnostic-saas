<template>
  <ClientOnly>
    <div v-if="mounted" class="digital-twin">
      <div ref="container" class="canvas-container">
        <canvas ref="canvas"></canvas>
        <div v-if="isLoading" class="loading-overlay">
          <div class="spinner"></div>
          <p>Loading...</p>
        </div>
        <div class="controls-hint">
          <div class="hint-title">⌨️ Keyboard</div>
          <div class="hint-grid"><span>W/S-Boom | A/D-Stick | Q/E-Bucket</span></div>
        </div>
      </div>
      <div class="control-panel">
        <h2>🎮 Controls</h2>
        <div class="keyboard-layout">
          <div class="key-row">
            <button class="key-btn" :class="{ active: pressedKeys.has('KeyQ') }" @mousedown="startMove('bucket', -1)"
              @mouseup="stopMove('bucket')"><span class="key-label">Q</span><span class="key-action">⟲</span></button>
            <button class="key-btn" :class="{ active: pressedKeys.has('KeyW') || pressedKeys.has('ArrowUp') }"
              @mousedown="startMove('boom', 1)" @mouseup="stopMove('boom')"><span class="key-label">W</span><span
                class="key-action">↑</span></button>
            <button class="key-btn" :class="{ active: pressedKeys.has('KeyE') }" @mousedown="startMove('bucket', 1)"
              @mouseup="stopMove('bucket')"><span class="key-label">E</span><span class="key-action">⟳</span></button>
          </div>
          <div class="key-row">
            <button class="key-btn" :class="{ active: pressedKeys.has('KeyA') || pressedKeys.has('ArrowLeft') }"
              @mousedown="startMove('stick', -1)" @mouseup="stopMove('stick')"><span class="key-label">A</span><span
                class="key-action">←</span></button>
            <button class="key-btn" :class="{ active: pressedKeys.has('KeyS') || pressedKeys.has('ArrowDown') }"
              @mousedown="startMove('boom', -1)" @mouseup="stopMove('boom')"><span class="key-label">S</span><span
                class="key-action">↓</span></button>
            <button class="key-btn" :class="{ active: pressedKeys.has('KeyD') || pressedKeys.has('ArrowRight') }"
              @mousedown="startMove('stick', 1)" @mouseup="stopMove('stick')"><span class="key-label">D</span><span
                class="key-action">→</span></button>
          </div>
        </div>
        <div class="values-display">
          <div class="value-item"><span class="value-label">🔧 Boom:</span><span class="value-number">{{
            equipment.cylinder_boom.position.toFixed(1) }}%</span></div>
          <div class="value-item"><span class="value-label">🔩 Stick:</span><span class="value-number">{{
            equipment.cylinder_stick.position.toFixed(1) }}%</span></div>
          <div class="value-item"><span class="value-label">🪣 Bucket:</span><span class="value-number">{{
            equipment.cylinder_bucket.position.toFixed(1) }}°</span></div>
        </div>
        <div class="quick-actions">
          <button @click="resetPosition" class="btn-secondary">🔄 Reset</button>
          <button @click="simulateFault" class="btn-danger">⚠️ Fault</button>
        </div>
      </div>
      <div class="sensor-dashboard">
        <h2>📊 Live Sensors</h2>
        <div class="sensor-grid">
          <div class="sensor-card" :class="{ fault: equipment.pump.fault }">
            <div class="sensor-header">
              <div class="sensor-icon">💧</div>
              <h3>Hydraulic Pump</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row"><span class="label">RPM:</span><span class="value">{{
                equipment.pump.speed_rpm.toFixed(0) }}</span></div>
              <div class="sensor-row"><span class="label">Pressure:</span><span class="value">{{
                equipment.pump.pressure_outlet.toFixed(1) }} bar</span></div>
              <div class="sensor-row"><span class="label">Temp:</span><span class="value"
                  :class="{ warning: equipment.pump.temperature > 75 }">{{ equipment.pump.temperature.toFixed(1)
                  }}°C</span>
              </div>
              <div class="sensor-row"><span class="label">Power:</span><span class="value">{{
                equipment.pump.power.toFixed(1) }} kW</span></div>
            </div>
          </div>
          <div class="sensor-card" :class="{ fault: equipment.cylinder_boom.fault }">
            <div class="sensor-header">
              <div class="sensor-icon">🔧</div>
              <h3>Boom Cylinder</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row"><span class="label">Position:</span><span class="value">{{
                equipment.cylinder_boom.position.toFixed(1) }}%</span></div>
              <div class="sensor-row"><span class="label">Pressure:</span><span class="value"
                  :class="{ warning: equipment.cylinder_boom.pressure > 200 }">{{
                    equipment.cylinder_boom.pressure.toFixed(1) }}
                  bar</span></div>
              <div class="sensor-row"><span class="label">Temp:</span><span class="value">{{
                equipment.cylinder_boom.temperature.toFixed(1) }}°C</span></div>
            </div>
            <div v-if="equipment.cylinder_boom.fault" class="fault-badge">⚠️ FAULT</div>
          </div>
          <div class="sensor-card">
            <div class="sensor-header">
              <div class="sensor-icon">🔩</div>
              <h3>Stick Cylinder</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row"><span class="label">Position:</span><span class="value">{{
                equipment.cylinder_stick.position.toFixed(1) }}%</span></div>
              <div class="sensor-row"><span class="label">Pressure:</span><span class="value">{{
                equipment.cylinder_stick.pressure.toFixed(1) }} bar</span></div>
              <div class="sensor-row"><span class="label">Temp:</span><span class="value">{{
                equipment.cylinder_stick.temperature.toFixed(1) }}°C</span></div>
            </div>
          </div>
          <div class="sensor-card">
            <div class="sensor-header">
              <div class="sensor-icon">🪣</div>
              <h3>Bucket Cylinder</h3>
            </div>
            <div class="sensor-values">
              <div class="sensor-row"><span class="label">Position:</span><span class="value">{{
                equipment.cylinder_bucket.position.toFixed(1) }}°</span></div>
              <div class="sensor-row"><span class="label">Pressure:</span><span class="value">{{
                equipment.cylinder_bucket.pressure.toFixed(1) }} bar</span></div>
              <div class="sensor-row"><span class="label">Temp:</span><span class="value">{{
                equipment.cylinder_bucket.temperature.toFixed(1) }}°C</span></div>
            </div>
          </div>
        </div>
      </div>
      <Transition name="slide-up">
        <div v-if="latestPrediction" class="prediction-panel"
          :class="latestPrediction.fault_detected ? 'fault' : 'normal'">
          <div class="prediction-content">
            <div class="status-icon">{{ latestPrediction.fault_detected ? '🔴' : '🟢' }}</div>
            <div class="prediction-details">
              <h3>{{ latestPrediction.fault_detected ? 'Fault Detected' : 'System Normal' }}</h3>
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
import { ref, reactive, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { useDigitalTwin } from '~/composables/useDigitalTwin'

const container = ref<HTMLDivElement>()
const canvas = ref<HTMLCanvasElement>()
const mounted = ref(false)
const isLoading = ref(true)

const { equipment, latestPrediction, updatePhysics, moveBoom, moveStick, moveBucket } = useDigitalTwin()
const pressedKeys = reactive(new Set<string>())
const activeControls = reactive({ boom: 0, stick: 0, bucket: 0 })

let scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  renderer: THREE.WebGLRenderer,
  controls: OrbitControls
let rootBone: THREE.Bone, baseBone: THREE.Bone, boomBone: THREE.Bone, stickBone: THREE.Bone, bucketBone: THREE.Bone
let baseMesh: THREE.Mesh, boomMesh: THREE.Mesh, stickMesh: THREE.Mesh, bucketMesh: THREE.Mesh
let cylinderLines: THREE.Line[] = []
let animationId: number, lastTime = Date.now()

onMounted(async () => {
  mounted.value = true
  await nextTick()
  if (!container.value || !canvas.value) return
  try {
    initScene()
    createBones()
    createMeshes()
    createCylinderLines()
    startAnim()
    setTimeout(() => (isLoading.value = false), 300)
  } catch (e) {
    console.error(e)
  }
})

function initScene() {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x1a1a2e)
  scene.fog = new THREE.Fog(0x1a1a2e, 20, 80)
  const w = container.value!.clientWidth,
    h = container.value!.clientHeight
  camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000)
  camera.position.set(15, 8, 15)
  camera.lookAt(0, 2, 0)
  renderer = new THREE.WebGLRenderer({ canvas: canvas.value!, antialias: true })
  renderer.setSize(w, h)
  renderer.shadowMap.enabled = true
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.target.set(0, 2, 0)
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const dl = new THREE.DirectionalLight(0xffffff, 1.2)
  dl.position.set(20, 30, 15)
  dl.castShadow = true
  scene.add(dl)
  scene.add(new THREE.HemisphereLight(0x87ceeb, 0x8b7355, 0.5))
  const g = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x2d3436 }))
  g.rotation.x = -Math.PI / 2
  g.receiveShadow = true
  scene.add(g)
  const gr = new THREE.GridHelper(80, 40, 0x636e72, 0x2d3436)
  gr.position.y = 0.01
  scene.add(gr)
  window.addEventListener('keydown', (e) => {
    pressedKeys.add(e.code)
    if (e.code === 'KeyW' || e.code === 'ArrowUp') activeControls.boom = 1
    if (e.code === 'KeyS' || e.code === 'ArrowDown') activeControls.boom = -1
    if (e.code === 'KeyA' || e.code === 'ArrowLeft') activeControls.stick = -1
    if (e.code === 'KeyD' || e.code === 'ArrowRight') activeControls.stick = 1
    if (e.code === 'KeyQ') activeControls.bucket = -1
    if (e.code === 'KeyE') activeControls.bucket = 1
  })
  window.addEventListener('keyup', (e) => {
    pressedKeys.delete(e.code)
    if (e.code === 'KeyW' || e.code === 'KeyS' || e.code === 'ArrowUp' || e.code === 'ArrowDown') activeControls.boom = 0
    if (e.code === 'KeyA' || e.code === 'KeyD' || e.code === 'ArrowLeft' || e.code === 'ArrowRight') activeControls.stick = 0
    if (e.code === 'KeyQ' || e.code === 'KeyE') activeControls.bucket = 0
  })
}

function createBones() {
  rootBone = new THREE.Bone()
  scene.add(rootBone)
  baseBone = new THREE.Bone()
  baseBone.position.set(0, 1, 0)
  rootBone.add(baseBone)
  boomBone = new THREE.Bone()
  boomBone.position.set(0.5, 1.2, 0)
  baseBone.add(boomBone)
  stickBone = new THREE.Bone()
  stickBone.position.set(4.5, 0, 0)
  boomBone.add(stickBone)
  bucketBone = new THREE.Bone()
  bucketBone.position.set(3.2, 0, 0)
  stickBone.add(bucketBone)
}

function createMeshes() {
  const y = 0xffb302,
    m = (c: number) => new THREE.MeshStandardMaterial({ color: c, metalness: 0.5, roughness: 0.6 })
  baseMesh = new THREE.Mesh(new THREE.BoxGeometry(4, 1.8, 3.5), m(y))
  baseMesh.position.set(0, -0.5, 0)
  baseMesh.castShadow = true
  baseBone.add(baseMesh)
  boomMesh = new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.6, 0.6), m(y))
  boomMesh.position.set(2.25, 0, 0)
  boomMesh.castShadow = true
  boomBone.add(boomMesh)
  stickMesh = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.5, 0.5), m(y))
  stickMesh.position.set(1.6, 0, 0)
  stickMesh.castShadow = true
  stickBone.add(stickMesh)
  bucketMesh = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.2, 1.5), m(0x555555))
  bucketMesh.position.set(0.75, 0, 0)
  bucketMesh.castShadow = true
  bucketBone.add(bucketMesh)
}

function createCylinderLines() {
  const mat = new THREE.LineBasicMaterial({ color: 0xff6b35, linewidth: 4 })
  for (let i = 0; i < 3; i++) {
    const pts = [new THREE.Vector3(), new THREE.Vector3()]
    const geo = new THREE.BufferGeometry().setFromPoints(pts)
    const line = new THREE.Line(geo, mat)
    scene.add(line)
    cylinderLines.push(line)
  }
}

function startAnim() {
  function anim() {
    animationId = requestAnimationFrame(anim)
    const now = Date.now(),
      dt = Math.min((now - lastTime) / 1000, 0.1)
    lastTime = now
    const s = 100 * dt

    if (activeControls.boom !== 0) moveBoom(Math.max(0, Math.min(100, equipment.cylinder_boom.position + activeControls.boom * s)))
    if (activeControls.stick !== 0) moveStick(Math.max(0, Math.min(100, equipment.cylinder_stick.position + activeControls.stick * s)))
    if (activeControls.bucket !== 0) moveBucket(Math.max(0, Math.min(100, equipment.cylinder_bucket.position + activeControls.bucket * s)))
    updatePhysics(dt)
    updatePose()
    // updateCylinderLines() // DISABLED
    controls.update()
    renderer.render(scene, camera)
  }
  anim()
}

function updatePose() {
  if (!boomBone) return
  boomBone.rotation.z = (equipment.cylinder_boom.position / 100) * (Math.PI / 3)
  stickBone.rotation.z = -(equipment.cylinder_stick.position / 100) * (Math.PI / 2.5)
  bucketBone.rotation.z = (equipment.cylinder_bucket.position / 100) * (Math.PI / 4)
  if (equipment.cylinder_boom.fault && boomMesh) {
    const mat = boomMesh.material as THREE.MeshStandardMaterial
    mat.emissive.setHex(0xff0000)
    mat.emissiveIntensity = 0.6
  } else if (boomMesh) (boomMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 0
}

// function updateCylinderLines() {
//   const p1 = new THREE.Vector3(), p2 = new THREE.Vector3()
//   if (cylinderLines[0]) {
//     baseBone.getWorldPosition(p1)
//     p1.add(new THREE.Vector3(0, 0.5, 0))
//     boomBone.getWorldPosition(p2)
//     p2.add(new THREE.Vector3(1.5, 0, 0))
//     const pts = [p1.clone(), p2.clone()]
//     cylinderLines[0].geometry.setFromPoints(pts)
//   }
//   if (cylinderLines[1]) {
//     boomBone.getWorldPosition(p1)
//     p1.add(new THREE.Vector3(2, -0.3, 0))
//     stickBone.getWorldPosition(p2)
//     const pts = [p1.clone(), p2.clone()]
//     cylinderLines[1].geometry.setFromPoints(pts)
//   }
//   if (cylinderLines[2]) {
//     stickBone.getWorldPosition(p1)
//     p1.add(new THREE.Vector3(0.8, -0.2, 0))
//     bucketBone.getWorldPosition(p2)
//     const pts = [p1.clone(), p2.clone()]
//     cylinderLines[2].geometry.setFromPoints(pts)
//   }
// }

function startMove(c: 'boom' | 'stick' | 'bucket', d: number) {
  activeControls[c] = d
}
function stopMove(c: 'boom' | 'stick' | 'bucket') {
  activeControls[c] = 0
}
function resetPosition() {
  activeControls.boom = 0
  activeControls.stick = 0
  activeControls.bucket = 0
  pressedKeys.clear()
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
    reasoning: 'Boom pressure exceeded 220 bar threshold'
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
  background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%)
}

.canvas-container {
  position: relative;
  grid-column: 1;
  grid-row: 1/3;
  border-radius: 16px;
  overflow: hidden;
  background: #0f0f1e;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5)
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
  z-index: 10
}

.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid rgba(255, 255, 255, 0.1);
  border-top-color: #ffb302;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 20px
}

@keyframes spin {
  to {
    transform: rotate(360deg)
  }
}

.controls-hint {
  position: absolute;
  top: 20px;
  left: 20px;
  background: rgba(0, 0, 0, 0.85);
  padding: 14px 18px;
  border-radius: 10px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(16, 185, 129, 0.3);
  z-index: 5
}

.hint-title {
  font-size: 15px;
  font-weight: 700;
  color: #10b981;
  margin-bottom: 6px
}

.hint-grid {
  display: grid;
  gap: 4px;
  font-size: 12px;
  color: #d1d5db
}

.control-panel {
  grid-column: 2;
  grid-row: 1;
  background: rgba(26, 26, 46, 0.95);
  border-radius: 16px;
  padding: 18px;
  color: white;
  overflow-y: auto
}

.control-panel h2 {
  margin: 0 0 16px 0;
  font-size: 18px;
  font-weight: 700
}

.keyboard-layout {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
  background: rgba(0, 0, 0, 0.3);
  padding: 16px;
  border-radius: 12px
}

.key-row {
  display: flex;
  gap: 8px;
  justify-content: center
}

.key-btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 14px 8px;
  background: linear-gradient(145deg, #2d2d3f, #1f1f2e);
  border: 2px solid #3d3d4f;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.12s;
  user-select: none;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3);
  min-width: 60px
}

.key-btn:hover {
  background: linear-gradient(145deg, #3d3d4f, #2d2d3f);
  border-color: #ffb302;
  transform: translateY(-1px);
  box-shadow: 0 5px 15px rgba(255, 179, 2, 0.25)
}

.key-btn.active {
  background: linear-gradient(145deg, #ffb302, #ff8c00) !important;
  border-color: #ffb302 !important;
  transform: translateY(1px) !important;
  box-shadow: 0 0 20px rgba(255, 179, 2, 0.6), inset 0 2px 8px rgba(0, 0, 0, 0.3) !important
}

.key-btn.active .key-label {
  color: #1f1f2e !important;
  text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3) !important
}

.key-btn.active .key-action {
  color: #1f1f2e !important
}

.key-label {
  font-size: 20px;
  font-weight: 700;
  color: #ffb302;
  font-family: 'Courier New', monospace;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
  transition: all 0.12s
}

.key-action {
  font-size: 16px;
  color: #9ca3af;
  transition: all 0.12s
}

.values-display {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  margin-bottom: 12px
}

.value-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 8px;
  background: rgba(16, 185, 129, 0.05);
  border-radius: 6px;
  border-left: 3px solid #10b981
}

.value-label {
  font-size: 13px;
  color: #9ca3af
}

.value-number {
  font-size: 15px;
  font-weight: 700;
  color: #10b981;
  font-family: 'Courier New', monospace
}

.quick-actions {
  display: flex;
  gap: 10px
}

.quick-actions button {
  flex: 1;
  padding: 12px;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s
}

.btn-secondary {
  background: linear-gradient(145deg, #374151, #1f2937);
  color: white;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3)
}

.btn-secondary:hover {
  background: linear-gradient(145deg, #4b5563, #374151);
  transform: translateY(-1px)
}

.btn-danger {
  background: linear-gradient(145deg, #ef4444, #dc2626);
  color: white;
  box-shadow: 0 3px 10px rgba(239, 68, 68, 0.4)
}

.btn-danger:hover {
  background: linear-gradient(145deg, #dc2626, #b91c1c);
  transform: translateY(-1px)
}

.sensor-dashboard {
  grid-column: 2;
  grid-row: 2;
  background: rgba(26, 26, 46, 0.95);
  border-radius: 16px;
  padding: 18px;
  color: white;
  max-height: 500px;
  overflow-y: auto
}

.sensor-dashboard h2 {
  margin: 0 0 14px 0;
  font-size: 16px;
  font-weight: 700
}

.sensor-grid {
  display: grid;
  gap: 10px
}

.sensor-card {
  background: #1f1f2e;
  border-radius: 10px;
  padding: 12px;
  border: 2px solid transparent;
  transition: all 0.3s
}

.sensor-card.fault {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
  animation: pulse-border 1s infinite
}

@keyframes pulse-border {

  0%,
  100% {
    border-color: #ef4444;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4)
  }

  50% {
    border-color: #dc2626;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.6)
  }
}

.sensor-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px
}

.sensor-icon {
  font-size: 22px
}

.sensor-header h3 {
  margin: 0;
  font-size: 13px;
  font-weight: 600
}

.sensor-values {
  font-size: 12px
}

.sensor-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05)
}

.sensor-row:last-child {
  border-bottom: none
}

.sensor-row .label {
  color: #9ca3af
}

.sensor-row .value {
  font-weight: 600;
  color: #10b981;
  font-family: 'Courier New', monospace
}

.sensor-row .value.warning {
  color: #f59e0b;
  animation: blink 1.5s infinite
}

@keyframes blink {

  0%,
  100% {
    opacity: 1
  }

  50% {
    opacity: 0.5
  }
}

.fault-badge {
  margin-top: 8px;
  padding: 6px;
  background: #ef4444;
  border-radius: 5px;
  font-weight: 700;
  font-size: 11px;
  text-align: center;
  animation: pulse 1s infinite
}

.prediction-panel {
  position: fixed;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.95);
  border-radius: 14px;
  padding: 18px 22px;
  color: white;
  min-width: 380px;
  max-width: 480px;
  backdrop-filter: blur(20px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7);
  z-index: 100
}

.prediction-panel.fault {
  border: 2px solid #ef4444;
  box-shadow: 0 20px 60px rgba(239, 68, 68, 0.5)
}

.prediction-panel.normal {
  border: 2px solid #10b981;
  box-shadow: 0 20px 60px rgba(16, 185, 129, 0.3)
}

.prediction-content {
  display: flex;
  gap: 14px;
  align-items: flex-start
}

.status-icon {
  font-size: 36px;
  flex-shrink: 0
}

.prediction-details h3 {
  margin: 0 0 8px 0;
  font-size: 17px;
  font-weight: 700
}

.confidence-text {
  font-size: 13px;
  font-weight: 600;
  color: #10b981;
  margin-bottom: 6px
}

.reasoning {
  font-size: 12px;
  color: #9ca3af;
  line-height: 1.5
}

.slide-up-enter-active,
.slide-up-leave-active {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1)
}

.slide-up-enter-from,
.slide-up-leave-to {
  transform: translateX(-50%) translateY(100px);
  opacity: 0
}

@keyframes pulse {

  0%,
  100% {
    opacity: 1
  }

  50% {
    opacity: 0.7
  }
}

::-webkit-scrollbar {
  width: 6px
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 3px
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 179, 2, 0.4);
  border-radius: 3px
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 179, 2, 0.6)
}
</style>
