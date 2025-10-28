<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <div class="container mx-auto px-4 py-8 space-y-6">
      <!-- Header -->
      <div class="flex items-center justify-between">
        <div>
          <h1 class="premium-heading-lg text-gray-900 dark:text-white">🔍 Diagnostics</h1>
          <p class="premium-body text-gray-600 dark:text-gray-300">
            Run automated diagnostics and analyze system health
          </p>
        </div>
        <PremiumButton @click="showRunModal = true" icon="heroicons:play">
          Run New Diagnostic
        </PremiumButton>
      </div>

      <!-- Active Sessions -->
      <div v-if="activeSessions.length > 0" class="space-y-4">
        <h2 class="premium-heading-sm text-gray-900 dark:text-white">🟢 Active Sessions</h2>
        <div class="grid gap-4">
          <div v-for="session in activeSessions" :key="session.id" class="premium-card p-6">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-4">
                <div class="h-3 w-3 bg-blue-500 rounded-full animate-pulse"></div>
                <div>
                  <p class="font-medium text-gray-900 dark:text-white">{{ session.name }}</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    {{ session.equipment }} • Started {{ session.startedAt }}
                  </p>
                </div>
              </div>
              <div class="flex items-center space-x-4">
                <div class="w-32">
                  <div class="flex items-center space-x-2">
                    <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        class="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        :style="{ width: session.progress + '%' }"
                      ></div>
                    </div>
                    <span class="text-sm font-medium text-gray-700 dark:text-gray-300"
                      >{{ session.progress }}%</span
                    >
                  </div>
                </div>
                <PremiumButton variant="secondary" size="sm" @click="cancelSession(session.id)">
                  Cancel
                </PremiumButton>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Recent Results -->
      <div class="premium-card">
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="premium-heading-sm text-gray-900 dark:text-white">
            📊 Recent Diagnostic Results
          </h3>
          <p class="premium-body text-gray-600 dark:text-gray-300">
            Completed diagnostic sessions and their findings
          </p>
        </div>
        <div class="p-6">
          <div class="space-y-4">
            <div
              v-for="result in recentResults"
              :key="result.id"
              class="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              <div class="flex items-center space-x-4">
                <div :class="`h-3 w-3 rounded-full ${getStatusColor(result.status)}`"></div>
                <div>
                  <p class="font-medium text-gray-900 dark:text-white">{{ result.name }}</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    {{ result.equipment }} • Completed {{ result.completedAt }}
                  </p>
                </div>
              </div>
              <div class="flex items-center space-x-4">
                <div class="text-right">
                  <p class="text-sm font-medium text-gray-900 dark:text-white">
                    {{ result.score }}/100
                  </p>
                  <p class="text-xs text-gray-500 dark:text-gray-400">Health Score</p>
                </div>
                <PremiumButton variant="secondary" size="sm" @click="viewResult(result.id)">
                  View Details
                </PremiumButton>
                <PremiumButton variant="secondary" size="sm" icon="heroicons:arrow-down-tray">
                  Export
                </PremiumButton>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Simple modal replacement (no hydration issues) -->
      <div
        v-if="showRunModal"
        class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
        @click="showRunModal = false"
      >
        <div class="premium-card max-w-md w-full m-4" @click.stop>
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white">🚀 Run New Diagnostic</h3>
            <p class="premium-body text-gray-600 dark:text-gray-300">
              Select equipment and diagnostic parameters
            </p>
          </div>

          <div class="p-6 space-y-4">
            <div class="space-y-2">
              <label class="premium-label">Equipment</label>
              <select v-model="selectedEquipment" class="premium-input">
                <option value="">Select equipment...</option>
                <option value="hyd-001">HYD-001 - Pump Station A</option>
                <option value="hyd-002">HYD-002 - Hydraulic Motor B</option>
                <option value="hyd-003">HYD-003 - Control Valve C</option>
              </select>
            </div>

            <div class="space-y-2">
              <label class="premium-label">Diagnostic Type</label>
              <select v-model="diagnosticType" class="premium-input">
                <option value="full">Full System Analysis</option>
                <option value="pressure">Pressure System Check</option>
                <option value="temperature">Temperature Analysis</option>
                <option value="vibration">Vibration Analysis</option>
              </select>
            </div>

            <div class="flex items-center space-x-2">
              <input
                id="email-notification"
                v-model="emailNotification"
                type="checkbox"
                class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label for="email-notification" class="text-sm text-gray-700 dark:text-gray-300">
                Send email notification when complete
              </label>
            </div>
          </div>

          <div class="p-6 border-t border-gray-200 dark:border-gray-700 flex space-x-3">
            <PremiumButton variant="secondary" @click="showRunModal = false" class="flex-1">
              Cancel
            </PremiumButton>
            <PremiumButton
              :disabled="!selectedEquipment"
              @click="startDiagnostic"
              class="flex-1"
              icon="heroicons:play"
            >
              Start Diagnostic
            </PremiumButton>
          </div>
        </div>
      </div>

      <!-- Results modal replacement -->
      <div
        v-if="showResultsModal"
        class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
        @click="showResultsModal = false"
      >
        <div class="premium-card max-w-4xl w-full m-4 max-h-[90vh] overflow-y-auto" @click.stop>
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 class="premium-heading-sm text-gray-900 dark:text-white">
              📊 Diagnostic Results: {{ selectedResult?.name }}
            </h3>
            <p class="premium-body text-gray-600 dark:text-gray-300">
              Detailed analysis and recommendations
            </p>
          </div>

          <div class="p-6 space-y-6">
            <!-- Summary Cards -->
            <div class="grid gap-4 md:grid-cols-3">
              <div class="premium-card p-4 text-center">
                <div class="text-2xl font-bold text-green-600 dark:text-green-400">
                  {{ selectedResult?.score }}/100
                </div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Overall Health Score</p>
              </div>
              <div class="premium-card p-4 text-center">
                <div class="text-2xl font-bold text-gray-900 dark:text-white">
                  {{ selectedResult?.issuesFound }}
                </div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Issues Found</p>
              </div>
              <div class="premium-card p-4 text-center">
                <div class="text-2xl font-bold text-gray-900 dark:text-white">
                  {{ selectedResult?.duration }}
                </div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Analysis Duration</p>
              </div>
            </div>

            <!-- Recommendations -->
            <div class="premium-card p-6">
              <h4 class="premium-heading-sm text-gray-900 dark:text-white mb-4">
                📝 Recommendations
              </h4>
              <div class="space-y-4">
                <div
                  class="p-4 border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg"
                >
                  <div class="flex items-start space-x-3">
                    <Icon
                      name="heroicons:exclamation-triangle"
                      class="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5"
                    />
                    <div>
                      <p class="font-medium text-yellow-800 dark:text-yellow-200">
                        Pressure System Maintenance
                      </p>
                      <p class="text-sm text-yellow-700 dark:text-yellow-300">
                        Schedule filter replacement within 2 weeks to prevent pressure fluctuations.
                      </p>
                      <p class="text-xs text-yellow-600 dark:text-yellow-400 mt-1">
                        Priority: Medium
                      </p>
                    </div>
                  </div>
                </div>

                <div
                  class="p-4 border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 rounded-lg"
                >
                  <div class="flex items-start space-x-3">
                    <Icon
                      name="heroicons:check-circle"
                      class="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5"
                    />
                    <div>
                      <p class="font-medium text-green-800 dark:text-green-200">
                        Temperature Monitoring
                      </p>
                      <p class="text-sm text-green-700 dark:text-green-300">
                        Temperature readings are within optimal range. Continue monitoring.
                      </p>
                      <p class="text-xs text-green-600 dark:text-green-400 mt-1">Status: Normal</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="p-6 border-t border-gray-200 dark:border-gray-700 flex space-x-3">
            <PremiumButton variant="secondary" @click="showResultsModal = false" class="flex-1">
              Close
            </PremiumButton>
            <PremiumButton class="flex-1" icon="heroicons:arrow-down-tray">
              Export PDF
            </PremiumButton>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Remove hydration-problematic imports
definePageMeta({
  title: 'Diagnostics | Hydraulic Diagnostic SaaS',
});

interface DiagnosticSession {
  id: number;
  name: string;
  equipment: string;
  startedAt: string;
  progress: number;
}

interface DiagnosticResult {
  id: number;
  name: string;
  equipment: string;
  completedAt: string;
  status: string;
  score: number;
  issuesFound: number;
  duration: string;
}

// Reactive state with proper types
const showRunModal = ref<boolean>(false);
const showResultsModal = ref<boolean>(false);
const selectedEquipment = ref<string>('');
const diagnosticType = ref<string>('full');
const priority = ref<string>('normal');
const emailNotification = ref<boolean>(true);
const selectedResult = ref<DiagnosticResult | null>(null);

// Demo data
const activeSessions = ref<DiagnosticSession[]>([
  {
    id: 1,
    name: 'Full System Analysis - HYD-001',
    equipment: 'HYD-001',
    startedAt: '2 minutes ago',
    progress: 65,
  },
]);

const recentResults = ref<DiagnosticResult[]>([
  {
    id: 1,
    name: 'Weekly Health Check',
    equipment: 'HYD-001',
    completedAt: '1 hour ago',
    status: 'completed',
    score: 92,
    issuesFound: 2,
    duration: '5 min',
  },
  {
    id: 2,
    name: 'Pressure System Analysis',
    equipment: 'HYD-002',
    completedAt: '3 hours ago',
    status: 'warning',
    score: 78,
    issuesFound: 4,
    duration: '8 min',
  },
  {
    id: 3,
    name: 'Vibration Analysis',
    equipment: 'HYD-003',
    completedAt: '1 day ago',
    status: 'completed',
    score: 96,
    issuesFound: 0,
    duration: '3 min',
  },
]);

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'bg-green-500';
    case 'warning':
      return 'bg-yellow-500';
    case 'error':
      return 'bg-red-500';
    default:
      return 'bg-gray-500';
  }
};

const startDiagnostic = (): void => {
  if (!selectedEquipment.value) return;

  // Simulate starting a diagnostic session
  const newSession: DiagnosticSession = {
    id: Date.now(),
    name: `${diagnosticType.value} - ${selectedEquipment.value.toUpperCase()}`,
    equipment: selectedEquipment.value.toUpperCase(),
    startedAt: 'Just now',
    progress: 0,
  };

  activeSessions.value.push(newSession);
  showRunModal.value = false;

  // Reset form
  selectedEquipment.value = '';
  diagnosticType.value = 'full';
  priority.value = 'normal';
  emailNotification.value = true;

  // Simulate progress for demo
  const interval = setInterval(() => {
    const session = activeSessions.value.find(s => s.id === newSession.id);
    if (session && session.progress < 100) {
      session.progress += Math.random() * 15;
    } else {
      clearInterval(interval);
      if (session) {
        session.progress = 100;
        // Move to completed after 1 second
        setTimeout(() => {
          const index = activeSessions.value.findIndex(s => s.id === newSession.id);
          if (index > -1) {
            activeSessions.value.splice(index, 1);
            // Add to results
            recentResults.value.unshift({
              id: Date.now(),
              name: newSession.name,
              equipment: newSession.equipment,
              completedAt: 'Just now',
              status: 'completed',
              score: Math.floor(Math.random() * 40) + 60,
              issuesFound: Math.floor(Math.random() * 5),
              duration: Math.floor(Math.random() * 8) + 2 + ' min',
            });
          }
        }, 1000);
      }
    }
  }, 800);
};

const cancelSession = (id: number): void => {
  const index = activeSessions.value.findIndex(session => session.id === id);
  if (index > -1) {
    activeSessions.value.splice(index, 1);
  }
};

const viewResult = (id: number): void => {
  selectedResult.value = recentResults.value.find(result => result.id === id) || null;
  showResultsModal.value = true;
};
</script>
