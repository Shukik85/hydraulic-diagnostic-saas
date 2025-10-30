<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="u-flex-between">
      <div>
        <h1 class="u-h2">Diagnostics Center</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          Run automated diagnostics and analyze system health
        </p>
      </div>
      <button @click="showRunModal = true" class="u-btn u-btn-primary u-btn-md">
        <Icon name="heroicons:play" class="w-4 h-4 mr-2" />
        Run New Diagnostic
      </button>
    </div>

    <!-- KPI Overview -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Active Sessions</h3>
          <div class="u-metric-icon bg-blue-100 dark:bg-blue-900/30">
            <Icon name="heroicons:play-circle" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <div class="u-metric-value">{{ activeSessions.length }}</div>
        <div class="u-metric-change text-gray-600 dark:text-gray-400 mt-2">
          <Icon name="heroicons:clock" class="w-4 h-4" />
          <span>Running now</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Success Rate</h3>
          <div class="u-metric-icon bg-green-100 dark:bg-green-900/30">
            <Icon name="heroicons:check-badge" class="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <div class="u-metric-value">98.5%</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>+2.1% this week</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Avg Duration</h3>
          <div class="u-metric-icon bg-purple-100 dark:bg-purple-900/30">
            <Icon name="heroicons:clock" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
          </div>
        </div>
        <div class="u-metric-value">4.2<span class="text-lg text-gray-500">min</span></div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-down" class="w-4 h-4" />
          <span>-0.8min faster</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Issues Found</h3>
          <div class="u-metric-icon bg-red-100 dark:bg-red-900/30">
            <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-red-600 dark:text-red-400" />
          </div>
        </div>
        <div class="u-metric-value">7</div>
        <div class="u-metric-change u-metric-change-negative mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>Needs attention</span>
        </div>
      </div>
    </div>

    <!-- Active Sessions -->
    <div v-if="activeSessions.length > 0">
      <h2 class="u-h4 mb-6">Active Diagnostic Sessions</h2>
      <div class="space-y-4">
        <div v-for="session in activeSessions" :key="session.id" class="u-card p-6">
          <div class="u-flex-between">
            <div class="flex items-center gap-4">
              <div class="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">{{ session.name }}</p>
                <p class="u-body-sm text-gray-500 dark:text-gray-400">
                  {{ session.equipment }} • Started {{ session.startedAt }}
                </p>
              </div>
            </div>
            <div class="flex items-center gap-6">
              <div class="w-32">
                <div class="flex items-center gap-2">
                  <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      class="bg-blue-500 h-2 rounded-full u-transition-fast"
                      :style="{ width: session.progress + '%' }"
                    ></div>
                  </div>
                  <span class="u-body-sm font-medium text-gray-700 dark:text-gray-300">
                    {{ Math.round(session.progress) }}%
                  </span>
                </div>
              </div>
              <button @click="cancelSession(session.id)" class="u-btn u-btn-ghost u-btn-sm">
                <Icon name="heroicons:x-mark" class="w-4 h-4 mr-1" />
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Results -->
    <div class="u-card">
      <div class="p-6 border-b border-gray-200 dark:border-gray-700">
        <div class="u-flex-between">
          <div>
            <h3 class="u-h4">Recent Diagnostic Results</h3>
            <p class="u-body text-gray-600 dark:text-gray-300 mt-1">
              Completed diagnostic sessions and their findings
            </p>
          </div>
          <div class="flex items-center gap-2">
            <select class="u-input text-sm py-1 px-2 w-40">
              <option>All Equipment</option>
              <option>HYD-001</option>
              <option>HYD-002</option>
              <option>HYD-003</option>
            </select>
            <button class="u-btn u-btn-secondary u-btn-sm">
              <Icon name="heroicons:funnel" class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
      
      <div class="p-6">
        <div class="overflow-x-auto">
          <table class="u-table">
            <thead>
              <tr>
                <th>Diagnostic Name</th>
                <th>Equipment</th>
                <th>Health Score</th>
                <th>Issues</th>
                <th>Completed</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="result in recentResults" :key="result.id">
                <td class="font-medium">{{ result.name }}</td>
                <td>{{ result.equipment }}</td>
                <td>
                  <div class="flex items-center gap-2">
                    <div class="w-12 h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                      <div 
                        class="h-2 rounded-full"
                        :class="result.score >= 90 ? 'bg-green-500' : result.score >= 70 ? 'bg-yellow-500' : 'bg-red-500'"
                        :style="{ width: result.score + '%' }"
                      ></div>
                    </div>
                    <span class="u-body-sm font-medium">{{ result.score }}/100</span>
                  </div>
                </td>
                <td>
                  <span 
                    class="u-badge"
                    :class="result.issuesFound === 0 ? 'u-badge-success' : result.issuesFound <= 2 ? 'u-badge-warning' : 'u-badge-error'"
                  >
                    {{ result.issuesFound }} issues
                  </span>
                </td>
                <td class="u-body-sm text-gray-500 dark:text-gray-400">{{ result.completedAt }}</td>
                <td>
                  <span 
                    class="u-badge"
                    :class="getStatusBadgeClass(result.status)"
                  >
                    <Icon :name="getStatusIcon(result.status)" class="w-3 h-3" />
                    {{ result.status }}
                  </span>
                </td>
                <td>
                  <div class="flex items-center gap-2">
                    <button @click="viewResult(result.id)" class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:eye" class="w-4 h-4" />
                    </button>
                    <button class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:arrow-down-tray" class="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Run Diagnostic Modal -->
    <div v-if="showRunModal" class="fixed inset-0 bg-black/50 z-50 u-flex-center" @click="showRunModal = false">
      <div class="u-card max-w-md w-full m-4" @click.stop>
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="u-h4">Run New Diagnostic</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Select equipment and diagnostic parameters
          </p>
        </div>

        <div class="p-6 space-y-4">
          <div>
            <label class="u-label">Equipment</label>
            <select v-model="selectedEquipment" class="u-input">
              <option value="">Select equipment...</option>
              <option value="hyd-001">HYD-001 - Pump Station A</option>
              <option value="hyd-002">HYD-002 - Hydraulic Motor B</option>
              <option value="hyd-003">HYD-003 - Control Valve C</option>
            </select>
          </div>

          <div>
            <label class="u-label">Diagnostic Type</label>
            <select v-model="diagnosticType" class="u-input">
              <option value="full">Full System Analysis</option>
              <option value="pressure">Pressure System Check</option>
              <option value="temperature">Temperature Analysis</option>
              <option value="vibration">Vibration Analysis</option>
            </select>
          </div>

          <div class="flex items-center gap-2">
            <input
              id="email-notification"
              v-model="emailNotification"
              type="checkbox"
              class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
            />
            <label for="email-notification" class="u-body text-gray-700 dark:text-gray-300">
              Send email notification when complete
            </label>
          </div>
        </div>

        <div class="p-6 border-t border-gray-200 dark:border-gray-700 flex gap-3">
          <button @click="showRunModal = false" class="u-btn u-btn-secondary flex-1">
            Cancel
          </button>
          <button
            :disabled="!selectedEquipment"
            @click="startDiagnostic"
            class="u-btn u-btn-primary flex-1"
          >
            <Icon name="heroicons:play" class="w-4 h-4 mr-2" />
            Start Diagnostic
          </button>
        </div>
      </div>
    </div>

    <!-- Results Modal -->
    <div v-if="showResultsModal" class="fixed inset-0 bg-black/50 z-50 u-flex-center" @click="showResultsModal = false">
      <div class="u-card max-w-4xl w-full m-4 max-h-[90vh] overflow-y-auto" @click.stop>
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="u-h4">Diagnostic Results: {{ selectedResult?.name }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Detailed analysis and recommendations
          </p>
        </div>

        <div class="p-6 space-y-6">
          <!-- Summary Cards -->
          <div class="grid gap-4 md:grid-cols-3">
            <div class="u-card p-4 text-center">
              <div class="text-2xl font-bold text-green-600 dark:text-green-400">
                {{ selectedResult?.score }}/100
              </div>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">Overall Health Score</p>
            </div>
            <div class="u-card p-4 text-center">
              <div class="text-2xl font-bold text-gray-900 dark:text-white">
                {{ selectedResult?.issuesFound }}
              </div>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">Issues Found</p>
            </div>
            <div class="u-card p-4 text-center">
              <div class="text-2xl font-bold text-gray-900 dark:text-white">
                {{ selectedResult?.duration }}
              </div>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">Analysis Duration</p>
            </div>
          </div>

          <!-- Recommendations -->
          <div class="u-card p-6">
            <h4 class="u-h5 mb-4">Recommendations</h4>
            <div class="space-y-4">
              <div class="p-4 border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                <div class="flex items-start gap-3">
                  <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                  <div>
                    <p class="font-medium text-yellow-800 dark:text-yellow-200">
                      Pressure System Maintenance
                    </p>
                    <p class="u-body-sm text-yellow-700 dark:text-yellow-300 mt-1">
                      Schedule filter replacement within 2 weeks to prevent pressure fluctuations.
                    </p>
                    <p class="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
                      Priority: Medium
                    </p>
                  </div>
                </div>
              </div>

              <div class="p-4 border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div class="flex items-start gap-3">
                  <Icon name="heroicons:check-circle" class="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                  <div>
                    <p class="font-medium text-green-800 dark:text-green-200">
                      Temperature Monitoring
                    </p>
                    <p class="u-body-sm text-green-700 dark:text-green-300 mt-1">
                      Temperature readings are within optimal range. Continue monitoring.
                    </p>
                    <p class="text-xs text-green-600 dark:text-green-400 mt-2">
                      Status: Normal
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="p-6 border-t border-gray-200 dark:border-gray-700 flex gap-3">
          <button @click="showResultsModal = false" class="u-btn u-btn-secondary flex-1">
            Close
          </button>
          <button class="u-btn u-btn-primary flex-1">
            <Icon name="heroicons:arrow-down-tray" class="w-4 h-4 mr-2" />
            Export PDF
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  title: 'Diagnostics Center',
  layout: 'default',
  middleware: ['auth']
})

interface DiagnosticSession {
  id: number
  name: string
  equipment: string
  startedAt: string
  progress: number
}

interface DiagnosticResult {
  id: number
  name: string
  equipment: string
  completedAt: string
  status: string
  score: number
  issuesFound: number
  duration: string
}

// Reactive state
const showRunModal = ref(false)
const showResultsModal = ref(false)
const selectedEquipment = ref('')
const diagnosticType = ref('full')
const emailNotification = ref(true)
const selectedResult = ref<DiagnosticResult | null>(null)

// Demo data
const activeSessions = ref<DiagnosticSession[]>([
  {
    id: 1,
    name: 'Full System Analysis - HYD-001',
    equipment: 'HYD-001',
    startedAt: '2 minutes ago',
    progress: 65
  }
])

const recentResults = ref<DiagnosticResult[]>([
  {
    id: 1,
    name: 'Weekly Health Check',
    equipment: 'HYD-001',
    completedAt: '1 hour ago',
    status: 'completed',
    score: 92,
    issuesFound: 2,
    duration: '5 min'
  },
  {
    id: 2,
    name: 'Pressure System Analysis',
    equipment: 'HYD-002',
    completedAt: '3 hours ago',
    status: 'warning',
    score: 78,
    issuesFound: 4,
    duration: '8 min'
  },
  {
    id: 3,
    name: 'Vibration Analysis',
    equipment: 'HYD-003',
    completedAt: '1 day ago',
    status: 'completed',
    score: 96,
    issuesFound: 0,
    duration: '3 min'
  }
])

const getStatusBadgeClass = (status: string) => {
  const classes = {
    'completed': 'u-badge-success',
    'warning': 'u-badge-warning',
    'error': 'u-badge-error',
    'processing': 'u-badge-processing'
  }
  return classes[status] || 'u-badge-info'
}

const getStatusIcon = (status: string) => {
  const icons = {
    'completed': 'heroicons:check-circle',
    'warning': 'heroicons:exclamation-triangle',
    'error': 'heroicons:x-circle',
    'processing': 'heroicons:clock'
  }
  return icons[status] || 'heroicons:clock'
}

const startDiagnostic = () => {
  if (!selectedEquipment.value) return

  const newSession: DiagnosticSession = {
    id: Date.now(),
    name: `${diagnosticType.value} - ${selectedEquipment.value.toUpperCase()}`,
    equipment: selectedEquipment.value.toUpperCase(),
    startedAt: 'Just now',
    progress: 0
  }

  activeSessions.value.push(newSession)
  showRunModal.value = false

  // Reset form
  selectedEquipment.value = ''
  diagnosticType.value = 'full'
  emailNotification.value = true

  // Simulate progress
  const interval = setInterval(() => {
    const session = activeSessions.value.find(s => s.id === newSession.id)
    if (session && session.progress < 100) {
      session.progress += Math.random() * 15
    } else {
      clearInterval(interval)
      if (session) {
        session.progress = 100
        setTimeout(() => {
          const index = activeSessions.value.findIndex(s => s.id === newSession.id)
          if (index > -1) {
            activeSessions.value.splice(index, 1)
            recentResults.value.unshift({
              id: Date.now(),
              name: newSession.name,
              equipment: newSession.equipment,
              completedAt: 'Just now',
              status: 'completed',
              score: Math.floor(Math.random() * 40) + 60,
              issuesFound: Math.floor(Math.random() * 5),
              duration: Math.floor(Math.random() * 8) + 2 + ' min'
            })
          }
        }, 1000)
      }
    }
  }, 800)
}

const cancelSession = (id: number) => {
  const index = activeSessions.value.findIndex(session => session.id === id)
  if (index > -1) {
    activeSessions.value.splice(index, 1)
  }
}

const viewResult = (id: number) => {
  selectedResult.value = recentResults.value.find(result => result.id === id) || null
  showResultsModal.value = true
}
</script>