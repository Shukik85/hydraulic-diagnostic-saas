<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="u-flex-between">
      <div>
        <h1 class="u-h2">Reports Center</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          Comprehensive analysis and recommendations for hydraulic systems
        </p>
      </div>
      <button class="u-btn u-btn-primary u-btn-md" @click="openGenerateModal = true">
        <Icon name="i-heroicons-plus" class="w-4 h-4 mr-2" />
        Generate Report
      </button>
    </div>

    <!-- Report Stats -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Total Reports</h3>
          <div class="u-metric-icon bg-blue-100 dark:bg-blue-900/30">
            <Icon name="i-heroicons-document-text" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <div class="u-metric-value">{{ reports.length }}</div>
        <div class="u-metric-change text-gray-600 dark:text-gray-400 mt-2">
          <Icon name="i-heroicons-arrow-trending-up" class="w-4 h-4" />
          <span>This month</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Critical Issues</h3>
          <div class="u-metric-icon bg-red-100 dark:bg-red-900/30">
            <Icon name="i-heroicons-exclamation-triangle" class="w-5 h-5 text-red-600 dark:text-red-400" />
          </div>
        </div>
        <div class="u-metric-value">{{ reports.filter(r => r.severity === 'critical' || r.severity === 'high').length }}</div>
        <div class="u-metric-change u-metric-change-negative mt-2">
          <Icon name="i-heroicons-arrow-trending-up" class="w-4 h-4" />
          <span>Requires attention</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Avg Resolution</h3>
          <div class="u-metric-icon bg-green-100 dark:bg-green-900/30">
            <Icon name="i-heroicons-clock" class="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <div class="u-metric-value">2.4<span class="text-lg text-gray-500">h</span></div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="i-heroicons-arrow-trending-down" class="w-4 h-4" />
          <span>-30min faster</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Success Rate</h3>
          <div class="u-metric-icon bg-purple-100 dark:bg-purple-900/30">
            <Icon name="i-heroicons-check-badge" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
          </div>
        </div>
        <div class="u-metric-value">94.7%</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="i-heroicons-arrow-trending-up" class="w-4 h-4" />
          <span>Quality improved</span>
        </div>
      </div>
    </div>

    <!-- Filters -->
    <div class="u-card p-6">
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="md:col-span-2">
          <label class="u-label">Search Reports</label>
          <div class="relative">
            <Icon name="i-heroicons-magnifying-glass" class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Report name or system..."
              class="u-input pl-10"
            />
          </div>
        </div>

        <div>
          <label class="u-label">Severity</label>
          <select v-model="selectedSeverity" class="u-input">
            <option value="all">All Levels</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>

        <div>
          <label class="u-label">Status</label>
          <select v-model="selectedStatus" class="u-input">
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="in_progress">In Progress</option>
            <option value="pending">Pending</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Reports Table -->
    <div class="u-card">
      <div class="p-6 border-b border-gray-200 dark:border-gray-700">
        <div class="u-flex-between">
          <h3 class="u-h4">Reports ({{ filteredReports.length }})</h3>
          <div class="flex items-center gap-2">
            <button class="u-btn u-btn-ghost u-btn-sm">
              <Icon name="i-heroicons-funnel" class="w-4 h-4" />
            </button>
            <button class="u-btn u-btn-ghost u-btn-sm">
              <Icon name="i-heroicons-arrow-down-tray" class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <div v-if="filteredReports.length === 0" class="p-12 text-center">
        <Icon name="i-heroicons-document-text" class="w-16 h-16 mx-auto text-gray-400 mb-4" />
        <h3 class="u-h5 text-gray-500 dark:text-gray-400 mb-2">No Reports Found</h3>
        <p class="u-body text-gray-400 dark:text-gray-500">Try adjusting your search filters or generate a new report</p>
      </div>

      <div v-else class="overflow-x-auto">
        <table class="u-table">
          <thead>
            <tr>
              <th>Report</th>
              <th>System</th>
              <th>Severity</th>
              <th>Status</th>
              <th>Created</th>
              <th>Completed</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="report in filteredReports" :key="report.id">
              <td>
                <div>
                  <p class="font-medium text-gray-900 dark:text-white">{{ report.title }}</p>
                  <p class="u-body-sm text-gray-500 dark:text-gray-400 line-clamp-1">{{ report.summary }}</p>
                </div>
              </td>
              <td>
                <div class="flex items-center gap-2">
                  <Icon name="i-heroicons-server" class="w-4 h-4 text-gray-400" />
                  <span class="u-body-sm">{{ report.system_name }}</span>
                </div>
              </td>
              <td>
                <span class="u-badge" :class="getSeverityBadgeClass(report.severity)">
                  {{ report.severity }}
                </span>
              </td>
              <td>
                <span class="u-badge" :class="getStatusBadgeClass(report.status)">
                  <Icon :name="getStatusIcon(report.status)" class="w-3 h-3" />
                  {{ report.status.replace('_', ' ') }}
                </span>
              </td>
              <td class="u-body-sm text-gray-500 dark:text-gray-400">
                {{ formatDateTime(report.created_at) }}
              </td>
              <td class="u-body-sm text-gray-500 dark:text-gray-400">
                {{ report.completed_at ? formatDateTime(report.completed_at) : '-' }}
              </td>
              <td>
                <div class="flex items-center gap-1">
                  <button @click="openReportModal(report)" class="u-btn u-btn-ghost u-btn-sm">
                    <Icon name="i-heroicons-eye" class="w-4 h-4" />
                  </button>
                  <button class="u-btn u-btn-ghost u-btn-sm">
                    <Icon name="i-heroicons-arrow-down-tray" class="w-4 h-4" />
                  </button>
                  <button class="u-btn u-btn-ghost u-btn-sm">
                    <Icon name="i-heroicons-ellipsis-horizontal" class="w-4 h-4" />
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Generate Report Modal Component -->
    <UReportGenerateModal 
      v-model="openGenerateModal" 
      :loading="generateLoading"
      @submit="onGenerate"
      @cancel="onCancelGenerate"
    />

    <!-- Report Details Modal -->
    <div v-if="showReportModal && selectedReport" class="fixed inset-0 bg-black/60 z-50 u-flex-center p-4" @click="closeReportModal">
      <div class="u-card max-w-4xl w-full max-h-[90vh] overflow-y-auto" @click.stop>
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <div class="u-flex-between">
            <div>
              <div class="flex items-center gap-3 mb-2">
                <span class="u-badge" :class="getSeverityBadgeClass(selectedReport.severity)">
                  {{ selectedReport.severity }}
                </span>
                <span class="u-badge" :class="getStatusBadgeClass(selectedReport.status)">
                  <Icon :name="getStatusIcon(selectedReport.status)" class="w-3 h-3" />
                  {{ selectedReport.status.replace('_', ' ') }}
                </span>
              </div>
              <h2 class="u-h3">{{ selectedReport.title }}</h2>
              <p class="u-body text-gray-500 dark:text-gray-400 mt-1">
                {{ selectedReport.system_name }} • {{ formatDateTime(selectedReport.created_at) }}
              </p>
            </div>
            <button @click="closeReportModal" class="u-btn u-btn-ghost u-btn-sm">
              <Icon name="i-heroicons-x-mark" class="w-5 h-5" />
            </button>
          </div>
        </div>

        <div class="p-6 space-y-6">
          <!-- Summary -->
          <div v-if="selectedReport.summary">
            <h3 class="u-h5 mb-3">Executive Summary</h3>
            <div class="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p class="u-body text-gray-700 dark:text-gray-300">{{ selectedReport.summary }}</p>
            </div>
          </div>

          <!-- Recommendations -->
          <div v-if="selectedReport.recommendations?.length">
            <h3 class="u-h5 mb-3">Action Items</h3>
            <div class="space-y-3">
              <div
                v-for="(recommendation, index) in selectedReport.recommendations"
                :key="index"
                class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800"
              >
                <div class="flex items-start gap-3">
                  <Icon name="i-heroicons-light-bulb" class="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
                  <p class="u-body text-gray-700 dark:text-gray-300">{{ recommendation }}</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Technical Details -->
          <div>
            <h3 class="u-h5 mb-3">Technical Details</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="u-body-sm text-gray-500 dark:text-gray-400 mb-1">Report ID</div>
                <div class="font-mono text-sm text-gray-900 dark:text-white">
                  #{{ selectedReport.id.toString().padStart(4, '0') }}
                </div>
              </div>
              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="u-body-sm text-gray-500 dark:text-gray-400 mb-1">System</div>
                <div class="u-body text-gray-900 dark:text-white">{{ selectedReport.system_name }}</div>
              </div>
              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="u-body-sm text-gray-500 dark:text-gray-400 mb-1">Created</div>
                <div class="u-body text-gray-900 dark:text-white">{{ formatDateTime(selectedReport.created_at) }}</div>
              </div>
              <div v-if="selectedReport.completed_at" class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div class="u-body-sm text-gray-500 dark:text-gray-400 mb-1">Completed</div>
                <div class="u-body text-gray-900 dark:text-white">{{ formatDateTime(selectedReport.completed_at) }}</div>
              </div>
            </div>
          </div>
        </div>

        <div class="p-6 border-t border-gray-200 dark:border-gray-700 flex gap-3">
          <button @click="closeReportModal" class="u-btn u-btn-secondary flex-1">
            Close
          </button>
          <button class="u-btn u-btn-primary flex-1">
            <Icon name="i-heroicons-arrow-down-tray" class="w-4 h-4 mr-2" />
            Download PDF
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  title: 'Reports Center',
  layout: 'dashboard',
  middleware: ['auth']
})

interface Report {
  id: number
  title: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  system_name: string
  created_at: string
  completed_at?: string
  summary?: string
  recommendations?: string[]
}

// State
const selectedReport = ref<Report | null>(null)
const showReportModal = ref(false)
const openGenerateModal = ref(false)
const generateLoading = ref(false)
const selectedSeverity = ref('all')
const selectedStatus = ref('all')
const searchQuery = ref('')

// Demo data (removed all emojis)
const reports = ref<Report[]>([
  {
    id: 1,
    title: 'HYD-001 Efficiency Analysis',
    severity: 'medium',
    status: 'completed',
    system_name: 'Pump Station A',
    created_at: '2024-10-24T10:30:00Z',
    completed_at: '2024-10-24T10:45:00Z',
    summary: 'System operating within normal parameters. Minor temperature deviations detected.',
    recommendations: [
      'Check cooling system',
      'Replace filter within a week', 
      'Calibrate temperature sensor'
    ]
  },
  {
    id: 2,
    title: 'HYD-002 Pressure Diagnostics',
    severity: 'high',
    status: 'completed',
    system_name: 'Hydraulic Motor B',
    created_at: '2024-10-24T09:15:00Z',
    completed_at: '2024-10-24T09:30:00Z',
    summary: 'Critical pressure fluctuations detected. Immediate intervention required.',
    recommendations: [
      'Stop system for inspection',
      'Check seal condition',
      'Replace pressure regulation valve'
    ]
  },
  {
    id: 3,
    title: 'HYD-003 Preventive Check',
    severity: 'low',
    status: 'in_progress',
    system_name: 'Control Valve C',
    created_at: '2024-10-24T08:00:00Z'
  },
  {
    id: 4,
    title: 'Weekly System Health Report',
    severity: 'low',
    status: 'completed',
    system_name: 'All Systems',
    created_at: '2024-10-21T06:00:00Z',
    completed_at: '2024-10-21T06:30:00Z',
    summary: 'Overall system health is good. No critical issues detected.',
    recommendations: ['Continue regular monitoring', 'Schedule next check in 7 days']
  }
])

// Computed
const filteredReports = computed(() => {
  return reports.value.filter(report => {
    const matchesSearch = !searchQuery.value || 
      report.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      report.system_name.toLowerCase().includes(searchQuery.value.toLowerCase())
    
    const matchesSeverity = selectedSeverity.value === 'all' || report.severity === selectedSeverity.value
    const matchesStatus = selectedStatus.value === 'all' || report.status === selectedStatus.value
    
    return matchesSearch && matchesSeverity && matchesStatus
  })
})

// Methods
const getSeverityBadgeClass = (severity: string) => {
  const classes = {
    'low': 'u-badge-success',
    'medium': 'u-badge-warning', 
    'high': 'u-badge-error',
    'critical': 'u-badge-error'
  }
  return classes[severity] || 'u-badge-info'
}

const getStatusBadgeClass = (status: string) => {
  const classes = {
    'completed': 'u-badge-success',
    'in_progress': 'u-badge-processing',
    'pending': 'u-badge-warning',
    'failed': 'u-badge-error'
  }
  return classes[status] || 'u-badge-info'
}

const getStatusIcon = (status: string) => {
  const icons = {
    'completed': 'i-heroicons-check-circle',
    'in_progress': 'i-heroicons-clock', 
    'pending': 'i-heroicons-pause-circle',
    'failed': 'i-heroicons-x-circle'
  }
  return icons[status] || 'i-heroicons-question-mark-circle'
}

const formatDateTime = (dateString: string) => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const openReportModal = (report: Report) => {
  selectedReport.value = report
  showReportModal.value = true
}

const closeReportModal = () => {
  selectedReport.value = null
  showReportModal.value = false
}

// FIXED: Now properly connected to UReportGenerateModal
const onGenerate = async (data: any) => {
  generateLoading.value = true
  try {
    console.log('Generating report with data:', data)
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    openGenerateModal.value = false
    
    // Show success notification
    alert('Report generation started! You will be notified when it is ready.')
    
  } catch (error: any) {
    console.error('Failed to generate report:', error)
    alert(`Failed to generate report: ${error?.message || 'Unknown error'}`)
  } finally {
    generateLoading.value = false
  }
}

const onCancelGenerate = () => {
  openGenerateModal.value = false
}

// ESC key handler for report details modal
onMounted(() => {
  const handleEsc = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && showReportModal.value) {
      closeReportModal()
    }
  }
  document.addEventListener('keydown', handleEsc)
  
  onUnmounted(() => {
    document.removeEventListener('keydown', handleEsc)
  })
})
</script>