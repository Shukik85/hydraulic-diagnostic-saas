<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Diagnostics</h1>
        <p class="text-muted-foreground">Run automated diagnostics and analyze system health</p>
      </div>
      <UiButton @click="showRunModal = true">
        <Icon name="lucide:play" class="mr-2 h-4 w-4" />
        Run New Diagnostic
      </UiButton>
    </div>

    <!-- Active Sessions -->
    <div v-if="activeSessions.length > 0" class="space-y-4">
      <h2 class="text-xl font-semibold">Active Sessions</h2>
      <div class="grid gap-4">
        <UiCard v-for="session in activeSessions" :key="session.id">
          <UiCardContent class="pt-6">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-4">
                <div class="h-3 w-3 bg-primary rounded-full animate-pulse"></div>
                <div>
                  <p class="font-medium">{{ session.name }}</p>
                  <p class="text-sm text-muted-foreground">{{ session.equipment }} • Started {{ session.startedAt }}</p>
                </div>
              </div>
              <div class="flex items-center space-x-2">
                <div class="w-32">
                  <div class="flex items-center space-x-2">
                    <div class="flex-1 bg-muted rounded-full h-2">
                      <div
                        class="bg-primary h-2 rounded-full transition-all duration-300"
                        :style="{ width: session.progress + '%' }"
                      ></div>
                    </div>
                    <span class="text-sm font-medium">{{ session.progress }}%</span>
                  </div>
                </div>
                <UiButton variant="outline" size="sm" @click="cancelSession(session.id)">
                  Cancel
                </UiButton>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </div>
    </div>

    <!-- Recent Results -->
    <UiCard>
      <UiCardHeader>
        <UiCardTitle>Recent Diagnostic Results</UiCardTitle>
        <UiCardDescription>Completed diagnostic sessions and their findings</UiCardDescription>
      </UiCardHeader>
      <UiCardContent>
        <div class="space-y-4">
          <div v-for="result in recentResults" :key="result.id" class="flex items-center justify-between p-4 border rounded-lg">
            <div class="flex items-center space-x-4">
              <div :class="`h-3 w-3 rounded-full ${getStatusColor(result.status)}`"></div>
              <div>
                <p class="font-medium">{{ result.name }}</p>
                <p class="text-sm text-muted-foreground">{{ result.equipment }} • Completed {{ result.completedAt }}</p>
              </div>
            </div>
            <div class="flex items-center space-x-4">
              <div class="text-right">
                <p class="text-sm font-medium">{{ result.score }}/100</p>
                <p class="text-xs text-muted-foreground">Health Score</p>
              </div>
              <UiButton variant="outline" size="sm" @click="viewResult(result.id)">
                View Details
              </UiButton>
              <UiButton variant="outline" size="sm">
                <Icon name="lucide:download" class="mr-2 h-4 w-4" />
                Export
              </UiButton>
            </div>
          </div>
        </div>
      </UiCardContent>
    </UiCard>

    <!-- Run Diagnostic Modal -->
    <UiDialog :open="showRunModal" @update:open="showRunModal = $event">
      <UiDialogContent>
        <UiDialogHeader>
          <UiDialogTitle>Run New Diagnostic Session</UiDialogTitle>
          <UiDialogDescription>
            Select equipment and diagnostic parameters
          </UiDialogDescription>
        </UiDialogHeader>

        <div class="space-y-4">
          <div class="space-y-2">
            <label class="text-sm font-medium">Equipment</label>
            <UiSelect v-model="selectedEquipment">
              <UiSelectItem value="hyd-001">HYD-001 - Pump Station A</UiSelectItem>
              <UiSelectItem value="hyd-002">HYD-002 - Hydraulic Motor B</UiSelectItem>
              <UiSelectItem value="hyd-003">HYD-003 - Control Valve C</UiSelectItem>
            </UiSelect>
          </div>

          <div class="space-y-2">
            <label class="text-sm font-medium">Diagnostic Type</label>
            <UiSelect v-model="diagnosticType">
              <UiSelectItem value="full">Full System Analysis</UiSelectItem>
              <UiSelectItem value="pressure">Pressure System Check</UiSelectItem>
              <UiSelectItem value="temperature">Temperature Analysis</UiSelectItem>
              <UiSelectItem value="vibration">Vibration Analysis</UiSelectItem>
            </UiSelect>
          </div>

          <div class="space-y-2">
            <label class="text-sm font-medium">Priority</label>
            <UiSelect v-model="priority">
              <UiSelectItem value="low">Low</UiSelectItem>
              <UiSelectItem value="normal">Normal</UiSelectItem>
              <UiSelectItem value="high">High</UiSelectItem>
            </UiSelect>
          </div>

          <div class="flex items-center space-x-2">
            <UiCheckbox id="email-notification" v-model="emailNotification" />
            <label for="email-notification" class="text-sm">Send email notification when complete</label>
          </div>
        </div>

        <UiDialogFooter>
          <UiButton variant="outline" @click="showRunModal = false">
            Cancel
          </UiButton>
          <UiButton @click="startDiagnostic" :disabled="!selectedEquipment">
            <Icon name="lucide:play" class="mr-2 h-4 w-4" />
            Start Diagnostic
          </UiButton>
        </UiDialogFooter>
      </UiDialogContent>
    </UiDialog>

    <!-- Results Detail Modal -->
    <UiDialog :open="showResultsModal" @update:open="showResultsModal = $event">
      <UiDialogContent class="max-w-4xl">
        <UiDialogHeader>
          <UiDialogTitle>Diagnostic Results: {{ selectedResult?.name }}</UiDialogTitle>
          <UiDialogDescription>
            Detailed analysis and recommendations
          </UiDialogDescription>
        </UiDialogHeader>

        <div class="space-y-6">
          <!-- Summary -->
          <div class="grid gap-4 md:grid-cols-3">
            <UiCard>
              <UiCardContent class="pt-6">
                <div class="text-center">
                  <div class="text-2xl font-bold text-status-success">{{ selectedResult?.score }}/100</div>
                  <p class="text-sm text-muted-foreground">Overall Health Score</p>
                </div>
              </UiCardContent>
            </UiCard>

            <UiCard>
              <UiCardContent class="pt-6">
                <div class="text-center">
                  <div class="text-2xl font-bold">{{ selectedResult?.issuesFound }}</div>
                  <p class="text-sm text-muted-foreground">Issues Found</p>
                </div>
              </UiCardContent>
            </UiCard>

            <UiCard>
              <UiCardContent class="pt-6">
                <div class="text-center">
                  <div class="text-2xl font-bold">{{ selectedResult?.duration }}</div>
                  <p class="text-sm text-muted-foreground">Analysis Duration</p>
                </div>
              </UiCardContent>
            </UiCard>
          </div>

          <!-- Charts -->
          <UiCard>
            <UiCardHeader>
              <UiCardTitle>Analysis Results</UiCardTitle>
            </UiCardHeader>
            <UiCardContent>
              <div class="h-[300px] w-full bg-muted/20 rounded-md flex items-center justify-center">
                <div class="text-center">
                  <Icon name="lucide:bar-chart-3" class="mx-auto h-12 w-12 text-muted-foreground mb-2" />
                  <p class="text-sm text-muted-foreground">Diagnostic charts will be rendered here</p>
                </div>
              </div>
            </UiCardContent>
          </UiCard>

          <!-- Recommendations -->
          <UiCard>
            <UiCardHeader>
              <UiCardTitle>Recommendations</UiCardTitle>
            </UiCardHeader>
            <UiCardContent>
              <div class="space-y-4">
                <div class="p-4 border rounded-lg">
                  <div class="flex items-start space-x-3">
                    <Icon name="lucide:alert-triangle" class="h-5 w-5 text-status-warning mt-0.5" />
                    <div>
                      <p class="font-medium">Pressure System Maintenance</p>
                      <p class="text-sm text-muted-foreground">Schedule filter replacement within 2 weeks to prevent pressure fluctuations.</p>
                      <p class="text-xs text-muted-foreground mt-1">Priority: Medium</p>
                    </div>
                  </div>
                </div>

                <div class="p-4 border rounded-lg">
                  <div class="flex items-start space-x-3">
                    <Icon name="lucide:check-circle" class="h-5 w-5 text-status-success mt-0.5" />
                    <div>
                      <p class="font-medium">Temperature Monitoring</p>
                      <p class="text-sm text-muted-foreground">Temperature readings are within optimal range. Continue monitoring.</p>
                      <p class="text-xs text-muted-foreground mt-1">Status: Normal</p>
                    </div>
                  </div>
                </div>
              </div>
            </UiCardContent>
          </UiCard>
        </div>

        <UiDialogFooter>
          <UiButton variant="outline" @click="showResultsModal = false">
            Close
          </UiButton>
          <UiButton>
            <Icon name="lucide:download" class="mr-2 h-4 w-4" />
            Export PDF
          </UiButton>
        </UiDialogFooter>
      </UiDialogContent>
    </UiDialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

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

const showRunModal = ref(false)
const showResultsModal = ref(false)
const selectedEquipment = ref('')
const diagnosticType = ref('full')
const priority = ref('normal')
const emailNotification = ref(true)
const selectedResult = ref<DiagnosticResult | null>(null)

const activeSessions = ref([
  {
    id: 1,
    name: 'Full System Analysis - HYD-001',
    equipment: 'HYD-001',
    startedAt: '2 minutes ago',
    progress: 65
  }
])

const recentResults = ref([
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

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'bg-status-success'
    case 'warning': return 'bg-status-warning'
    case 'error': return 'bg-status-error'
    default: return 'bg-muted-foreground'
  }
}

const startDiagnostic = () => {
  // Simulate starting a diagnostic session
  const newSession = {
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
  priority.value = 'normal'
  emailNotification.value = true
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
