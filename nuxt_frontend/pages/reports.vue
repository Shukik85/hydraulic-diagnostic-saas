<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Reports</h1>
        <p class="text-muted-foreground">Generate and manage diagnostic reports</p>
      </div>
      <UiButton @click="showGenerateModal = true">
        <Icon name="lucide:plus" class="mr-2 h-4 w-4" />
        Generate Report
      </UiButton>
    </div>

    <!-- Filters -->
    <UiCard>
      <UiCardContent class="pt-6">
        <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div class="flex flex-1 items-center gap-2">
            <div class="relative flex-1 max-w-sm">
              <Icon name="lucide:search" class="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <UiInput
                placeholder="Search reports..."
                class="pl-10"
                v-model="searchQuery"
              />
            </div>
            <UiSelect v-model="typeFilter">
              <UiSelectItem value="all">All Types</UiSelectItem>
              <UiSelectItem value="diagnostic">Diagnostic</UiSelectItem>
              <UiSelectItem value="maintenance">Maintenance</UiSelectItem>
              <UiSelectItem value="performance">Performance</UiSelectItem>
            </UiSelect>
            <UiSelect v-model="dateFilter">
              <UiSelectItem value="all">All Dates</UiSelectItem>
              <UiSelectItem value="today">Today</UiSelectItem>
              <UiSelectItem value="week">This Week</UiSelectItem>
              <UiSelectItem value="month">This Month</UiSelectItem>
            </UiSelect>
          </div>
          <div class="flex items-center gap-2">
            <UiButton variant="outline" size="sm">
              <Icon name="lucide:filter" class="mr-2 h-4 w-4" />
              More Filters
            </UiButton>
          </div>
        </div>
      </UiCardContent>
    </UiCard>

    <!-- Reports Grid -->
    <div class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      <UiCard
        v-for="report in filteredReports"
        :key="report.id"
        class="cursor-pointer hover:shadow-lg transition-shadow"
        @click="previewReport(report)"
      >
        <UiCardHeader>
          <div class="flex items-center justify-between">
            <UiCardTitle class="text-lg">{{ report.title }}</UiCardTitle>
            <div :class="`h-3 w-3 rounded-full ${getStatusColor(report.status)}`"></div>
          </div>
          <UiCardDescription>{{ report.type }} • {{ report.equipment }}</UiCardDescription>
        </UiCardHeader>
        <UiCardContent>
          <div class="space-y-4">
            <p class="text-sm text-muted-foreground line-clamp-2">{{ report.description }}</p>
            <div class="flex items-center justify-between text-sm">
              <span class="text-muted-foreground">Generated</span>
              <span>{{ report.generatedAt }}</span>
            </div>
            <div class="flex items-center justify-between text-sm">
              <span class="text-muted-foreground">Size</span>
              <span>{{ report.size }}</span>
            </div>
          </div>
        </UiCardContent>
        <UiCardFooter>
          <div class="flex items-center justify-between w-full">
            <UiButton variant="outline" size="sm" @click.stop="downloadReport(report)">
              <Icon name="lucide:download" class="mr-2 h-4 w-4" />
              Download
            </UiButton>
            <UiButton variant="outline" size="sm" @click.stop="shareReport(report)">
              <Icon name="lucide:share" class="mr-2 h-4 w-4" />
              Share
            </UiButton>
          </div>
        </UiCardFooter>
      </UiCard>
    </div>

    <!-- Generate Report Modal -->
    <UiDialog :open="showGenerateModal" @update:open="showGenerateModal = $event">
      <UiDialogContent>
        <UiDialogHeader>
          <UiDialogTitle>Generate New Report</UiDialogTitle>
          <UiDialogDescription>
            Select report type and parameters
          </UiDialogDescription>
        </UiDialogHeader>

        <div class="space-y-4">
          <div class="space-y-2">
            <label class="text-sm font-medium">Report Type</label>
            <UiSelect v-model="reportType">
              <UiSelectItem value="diagnostic">Diagnostic Report</UiSelectItem>
              <UiSelectItem value="maintenance">Maintenance Report</UiSelectItem>
              <UiSelectItem value="performance">Performance Analysis</UiSelectItem>
              <UiSelectItem value="compliance">Compliance Report</UiSelectItem>
            </UiSelect>
          </div>

          <div class="space-y-2">
            <label class="text-sm font-medium">Equipment</label>
            <UiSelect v-model="selectedEquipment">
              <UiSelectItem value="all">All Equipment</UiSelectItem>
              <UiSelectItem value="hyd-001">HYD-001 - Pump Station A</UiSelectItem>
              <UiSelectItem value="hyd-002">HYD-002 - Hydraulic Motor B</UiSelectItem>
              <UiSelectItem value="hyd-003">HYD-003 - Control Valve C</UiSelectItem>
            </UiSelect>
          </div>

          <div class="space-y-2">
            <label class="text-sm font-medium">Date Range</label>
            <UiSelect v-model="dateRange">
              <UiSelectItem value="7d">Last 7 Days</UiSelectItem>
              <UiSelectItem value="30d">Last 30 Days</UiSelectItem>
              <UiSelectItem value="90d">Last 90 Days</UiSelectItem>
              <UiSelectItem value="custom">Custom Range</UiSelectItem>
            </UiSelect>
          </div>

          <div class="space-y-2">
            <label class="text-sm font-medium">Format</label>
            <div class="flex gap-2">
              <UiButton
                variant="outline"
                size="sm"
                :class="{ 'bg-muted': selectedFormat === 'pdf' }"
                @click="selectedFormat = 'pdf'"
              >
                PDF
              </UiButton>
              <UiButton
                variant="outline"
                size="sm"
                :class="{ 'bg-muted': selectedFormat === 'excel' }"
                @click="selectedFormat = 'excel'"
              >
                Excel
              </UiButton>
              <UiButton
                variant="outline"
                size="sm"
                :class="{ 'bg-muted': selectedFormat === 'csv' }"
                @click="selectedFormat = 'csv'"
              >
                CSV
              </UiButton>
            </div>
          </div>

          <div class="flex items-center space-x-2">
            <UiCheckbox id="email-report" v-model="emailReport" />
            <label for="email-report" class="text-sm">Email report when ready</label>
          </div>

          <div class="flex items-center space-x-2">
            <UiCheckbox id="schedule-report" v-model="scheduleReport" />
            <label for="schedule-report" class="text-sm">Schedule recurring report</label>
          </div>
        </div>

        <UiDialogFooter>
          <UiButton variant="outline" @click="showGenerateModal = false">
            Cancel
          </UiButton>
          <UiButton @click="generateReport">
            <Icon name="lucide:file-text" class="mr-2 h-4 w-4" />
            Generate Report
          </UiButton>
        </UiDialogFooter>
      </UiDialogContent>
    </UiDialog>

    <!-- Preview Modal -->
    <UiDialog :open="showPreviewModal" @update:open="showPreviewModal = $event">
      <UiDialogContent class="max-w-4xl max-h-[80vh] overflow-y-auto">
        <UiDialogHeader>
          <UiDialogTitle>{{ selectedReport?.title }}</UiDialogTitle>
          <UiDialogDescription>
            Report preview and details
          </UiDialogDescription>
        </UiDialogHeader>

        <div class="space-y-6">
          <!-- Report Preview -->
          <div class="border rounded-lg p-6 bg-muted/20">
            <div class="text-center py-12">
              <Icon name="lucide:file-text" class="mx-auto h-16 w-16 text-muted-foreground mb-4" />
              <h3 class="text-lg font-medium mb-2">Report Preview</h3>
              <p class="text-muted-foreground mb-4">
                This is where the PDF preview would be displayed
              </p>
              <div class="text-sm text-muted-foreground">
                <p><strong>Type:</strong> {{ selectedReport?.type }}</p>
                <p><strong>Equipment:</strong> {{ selectedReport?.equipment }}</p>
                <p><strong>Generated:</strong> {{ selectedReport?.generatedAt }}</p>
                <p><strong>Size:</strong> {{ selectedReport?.size }}</p>
              </div>
            </div>
          </div>

          <!-- Report Summary -->
          <UiCard>
            <UiCardHeader>
              <UiCardTitle>Report Summary</UiCardTitle>
            </UiCardHeader>
            <UiCardContent>
              <div class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <p class="text-sm text-muted-foreground">Total Pages</p>
                    <p class="text-lg font-medium">12</p>
                  </div>
                  <div>
                    <p class="text-sm text-muted-foreground">Data Points</p>
                    <p class="text-lg font-medium">1,247</p>
                  </div>
                </div>
                <div>
                  <p class="text-sm text-muted-foreground mb-2">Key Findings</p>
                  <ul class="text-sm space-y-1">
                    <li>• System health score: 92/100</li>
                    <li>• 3 maintenance items identified</li>
                    <li>• Pressure stability within acceptable range</li>
                    <li>• Filter replacement recommended</li>
                  </ul>
                </div>
              </div>
            </UiCardContent>
          </UiCard>
        </div>

        <UiDialogFooter>
          <UiButton variant="outline" @click="showPreviewModal = false">
            Close
          </UiButton>
          <UiButton @click="downloadReport(selectedReport)">
            <Icon name="lucide:download" class="mr-2 h-4 w-4" />
            Download PDF
          </UiButton>
        </UiDialogFooter>
      </UiDialogContent>
    </UiDialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Report {
  id: number
  title: string
  type: string
  equipment: string
  description: string
  status: string
  generatedAt: string
  size: string
}

const showGenerateModal = ref(false)
const showPreviewModal = ref(false)
const searchQuery = ref('')
const typeFilter = ref('all')
const dateFilter = ref('all')
const reportType = ref('diagnostic')
const selectedEquipment = ref('all')
const dateRange = ref('30d')
const selectedFormat = ref('pdf')
const emailReport = ref(true)
const scheduleReport = ref(false)
const selectedReport = ref<Report | null>(null)

const reports = ref([
  {
    id: 1,
    title: 'Monthly Diagnostic Report',
    type: 'Diagnostic',
    equipment: 'HYD-001',
    description: 'Comprehensive health analysis of Pump Station A including pressure, temperature, and vibration data.',
    status: 'completed',
    generatedAt: '2024-01-20',
    size: '2.4 MB'
  },
  {
    id: 2,
    title: 'Maintenance Schedule',
    type: 'Maintenance',
    equipment: 'All Systems',
    description: 'Upcoming maintenance tasks and service recommendations for all equipment.',
    status: 'completed',
    generatedAt: '2024-01-18',
    size: '1.8 MB'
  },
  {
    id: 3,
    title: 'Performance Analysis Q4',
    type: 'Performance',
    equipment: 'HYD-002',
    description: 'Quarterly performance metrics and efficiency analysis for Hydraulic Motor B.',
    status: 'processing',
    generatedAt: '2024-01-15',
    size: '3.1 MB'
  },
  {
    id: 4,
    title: 'Compliance Audit',
    type: 'Compliance',
    equipment: 'All Systems',
    description: 'Regulatory compliance check and safety certification report.',
    status: 'completed',
    generatedAt: '2024-01-10',
    size: '4.2 MB'
  }
])

const filteredReports = computed(() => {
  return reports.value.filter(report => {
    const matchesSearch = report.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
                         report.description.toLowerCase().includes(searchQuery.value.toLowerCase())

    const matchesType = typeFilter.value === 'all' || report.type.toLowerCase() === typeFilter.value

    return matchesSearch && matchesType
  })
})

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'bg-status-success'
    case 'processing': return 'bg-status-warning'
    case 'failed': return 'bg-status-error'
    default: return 'bg-muted-foreground'
  }
}

const generateReport = () => {
  // Simulate report generation
  const newReport = {
    id: Date.now(),
    title: `${reportType.value} Report`,
    type: reportType.value.charAt(0).toUpperCase() + reportType.value.slice(1),
    equipment: selectedEquipment.value === 'all' ? 'All Systems' : selectedEquipment.value.toUpperCase(),
    description: `Generated ${reportType.value} report for ${selectedEquipment.value === 'all' ? 'all equipment' : selectedEquipment.value.toUpperCase()}`,
    status: 'processing',
    generatedAt: new Date().toLocaleDateString(),
    size: '1.5 MB'
  }

  reports.value.unshift(newReport)
  showGenerateModal.value = false

  // Reset form
  reportType.value = 'diagnostic'
  selectedEquipment.value = 'all'
  dateRange.value = '30d'
  selectedFormat.value = 'pdf'
  emailReport.value = true
  scheduleReport.value = false
}

const previewReport = (report: Report) => {
  selectedReport.value = report
  showPreviewModal.value = true
}

const downloadReport = (report: Report | null) => {
  // Simulate download
  if (report) {
    console.log('Downloading report:', report.title)
  }
}

const shareReport = (report: Report) => {
  // Simulate sharing
  console.log('Sharing report:', report.title)
}
</script>
