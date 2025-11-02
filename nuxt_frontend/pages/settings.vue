<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h1 class="u-h2">{{ t('settings.title') }}</h1>
      <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.subtitle') }}</p>
    </div>

    <!-- Settings Navigation -->
    <div class="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        @click="activeTab = tab.id"
        class="px-4 py-2 text-sm font-medium rounded-md u-transition-fast"
        :class="activeTab === tab.id ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'"
      >
        <Icon :name="tab.icon" class="w-4 h-4 mr-2 inline" />
        {{ tab.name }}
      </button>
    </div>

    <!-- Profile Settings -->
    <div v-if="activeTab === 'profile'" class="space-y-6">
      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">{{ t('settings.profile.title') }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.profile.subtitle') }}</p>
        </div>
        
        <div class="space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="u-label">{{ t('settings.profile.firstName') }}</label>
              <input v-model="profile.firstName" class="u-input" />
            </div>
            <div>
              <label class="u-label">{{ t('settings.profile.lastName') }}</label>
              <input v-model="profile.lastName" class="u-input" />
            </div>
          </div>
          
          <div>
            <label class="u-label">{{ t('settings.profile.emailAddress') }}</label>
            <input v-model="profile.email" type="email" class="u-input" />
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="u-label">{{ t('settings.profile.company') }}</label>
              <input v-model="profile.company" class="u-input" />
            </div>
            <div>
              <label class="u-label">{{ t('settings.profile.phone') }}</label>
              <input v-model="profile.phone" type="tel" class="u-input" />
            </div>
          </div>
          
          <div class="pt-4">
            <button class="u-btn u-btn-primary u-btn-md">
              <Icon name="heroicons:check" class="w-4 h-4 mr-2" />
              {{ t('settings.profile.saveChanges') }}
            </button>
          </div>
        </div>
      </div>

      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">{{ t('settings.profile.changePassword') }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.profile.changePasswordSubtitle') }}</p>
        </div>
        
        <div class="space-y-4">
          <div>
            <label class="u-label">{{ t('settings.profile.currentPassword') }}</label>
            <input type="password" class="u-input" :placeholder="t('settings.profile.currentPasswordPlaceholder')" />
          </div>
          <div>
            <label class="u-label">{{ t('settings.profile.newPassword') }}</label>
            <input type="password" class="u-input" :placeholder="t('settings.profile.newPasswordPlaceholder')" />
          </div>
          <div>
            <label class="u-label">{{ t('settings.profile.confirmPassword') }}</label>
            <input type="password" class="u-input" :placeholder="t('settings.profile.confirmPasswordPlaceholder')" />
          </div>
          
          <div class="pt-4">
            <button class="u-btn u-btn-primary u-btn-md">
              <Icon name="heroicons:key" class="w-4 h-4 mr-2" />
              {{ t('settings.profile.updatePassword') }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Notifications Settings -->
    <div v-if="activeTab === 'notifications'" class="space-y-6">
      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">{{ t('settings.notifications.title') }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.notifications.subtitle') }}</p>
        </div>
        
        <div class="space-y-6">
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.notifications.systemAlerts') }}</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.notifications.systemAlertsDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.systemAlerts" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.notifications.maintenanceReminders') }}</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.notifications.maintenanceRemindersDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.maintenanceReminders" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.notifications.diagnosticReports') }}</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.notifications.diagnosticReportsDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.diagnosticReports" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.notifications.weeklySummary') }}</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.notifications.weeklySummaryDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.weeklySummary" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
        </div>
      </div>
    </div>

    <!-- Integrations Settings -->
    <div v-if="activeTab === 'integrations'" class="space-y-6">
      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">{{ t('settings.integrations.title') }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.integrations.subtitle') }}</p>
        </div>
        
        <div class="space-y-4">
          <div class="u-flex-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg u-flex-center">
                <Icon name="heroicons:cloud" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.integrations.scadaSystem') }}</p>
                <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.integrations.scadaSystemDesc') }}</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span class="u-badge u-badge-success">{{ t('settings.integrations.connected') }}</span>
              <button class="u-btn u-btn-secondary u-btn-sm">{{ t('settings.integrations.configure') }}</button>
            </div>
          </div>

          <div class="u-flex-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg u-flex-center">
                <Icon name="heroicons:circle-stack" class="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.integrations.erpSystem') }}</p>
                <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.integrations.erpSystemDesc') }}</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span class="u-badge u-badge-warning">{{ t('settings.integrations.notConnected') }}</span>
              <button class="u-btn u-btn-primary u-btn-sm">{{ t('settings.integrations.connect') }}</button>
            </div>
          </div>

          <div class="u-flex-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg u-flex-center">
                <Icon name="heroicons:chat-bubble-left-right" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.integrations.slackIntegration') }}</p>
                <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.integrations.slackIntegrationDesc') }}</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span class="u-badge u-badge-warning">{{ t('settings.integrations.notConnected') }}</span>
              <button class="u-btn u-btn-primary u-btn-sm">{{ t('settings.integrations.connect') }}</button>
            </div>
          </div>
        </div>
      </div>

      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">{{ t('settings.integrations.apiKeys') }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.integrations.apiKeysSubtitle') }}</p>
        </div>
        
        <div class="space-y-4">
          <div class="u-flex-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.integrations.primaryApiKey') }}</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400 font-mono">hds_sk_...aBcD1234</p>
            </div>
            <div class="flex items-center gap-2">
              <button class="u-btn u-btn-secondary u-btn-sm">
                <Icon name="heroicons:clipboard" class="w-4 h-4 mr-1" />
                {{ t('settings.integrations.copy') }}
              </button>
              <button class="u-btn u-btn-ghost u-btn-sm">
                <Icon name="heroicons:arrow-path" class="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <button class="u-btn u-btn-ghost u-btn-md">
            <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
            {{ t('settings.integrations.generateNewKey') }}
          </button>
        </div>
      </div>
    </div>

    <!-- Team Settings -->
    <div v-if="activeTab === 'team'" class="space-y-6">
      <div class="u-flex-between">
        <div>
          <h3 class="u-h4">{{ t('settings.team.title') }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.team.subtitle') }}</p>
        </div>
        <button class="u-btn u-btn-primary u-btn-md">
          <Icon name="heroicons:user-plus" class="w-4 h-4 mr-2" />
          {{ t('settings.team.inviteUser') }}
        </button>
      </div>

      <div class="u-card">
        <div class="overflow-x-auto">
          <table class="u-table">
            <thead>
              <tr>
                <th>{{ t('settings.team.user') }}</th>
                <th>{{ t('settings.team.role') }}</th>
                <th>{{ t('settings.team.lastActive') }}</th>
                <th>{{ t('settings.team.status') }}</th>
                <th>{{ t('settings.team.actions') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <div class="flex items-center gap-3">
                    <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full u-flex-center">
                      <Icon name="heroicons:user" class="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <p class="font-medium text-gray-900 dark:text-white">John Doe</p>
                      <p class="u-body-sm text-gray-500 dark:text-gray-400">john.doe@company.com</p>
                    </div>
                  </div>
                </td>
                <td><span class="u-badge u-badge-info">{{ t('settings.team.admin') }}</span></td>
                <td class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.team.hoursAgo', ['2']) }}</td>
                <td><span class="u-badge u-badge-success">{{ t('settings.team.active') }}</span></td>
                <td>
                  <div class="flex items-center gap-1">
                    <button class="u-btn u-btn-ghost u-btn-sm"><Icon name="heroicons:pencil" class="w-4 h-4" /></button>
                    <button class="u-btn u-btn-ghost u-btn-sm"><Icon name="heroicons:ellipsis-horizontal" class="w-4 h-4" /></button>
                  </div>
                </td>
              </tr>
              
              <tr>
                <td>
                  <div class="flex items-center gap-3">
                    <div class="w-8 h-8 bg-gradient-to-br from-green-500 to-teal-500 rounded-full u-flex-center">
                      <Icon name="heroicons:user" class="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <p class="font-medium text-gray-900 dark:text-white">Jane Smith</p>
                      <p class="u-body-sm text-gray-500 dark:text-gray-400">jane.smith@company.com</p>
                    </div>
                  </div>
                </td>
                <td><span class="u-badge u-badge-warning">{{ t('settings.team.engineer') }}</span></td>
                <td class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.team.dayAgo', ['1']) }}</td>
                <td><span class="u-badge u-badge-success">{{ t('settings.team.active') }}</span></td>
                <td>
                  <div class="flex items-center gap-1">
                    <button class="u-btn u-btn-ghost u-btn-sm"><Icon name="heroicons:pencil" class="w-4 h-4" /></button>
                    <button class="u-btn u-btn-ghost u-btn-sm"><Icon name="heroicons:ellipsis-horizontal" class="w-4 h-4" /></button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- System Settings -->
    <div v-if="activeTab === 'system'" class="space-y-6">
      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">{{ t('settings.system.title') }}</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('settings.system.subtitle') }}</p>
        </div>
        
        <div class="space-y-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="u-label">{{ t('settings.system.defaultAlertThreshold') }}</label>
              <select class="u-input">
                <option>{{ t('settings.system.warningAt75') }}</option>
                <option>{{ t('settings.system.warningAt80') }}</option>
                <option>{{ t('settings.system.warningAt90') }}</option>
              </select>
            </div>
            <div>
              <label class="u-label">{{ t('settings.system.dataRetentionPeriod') }}</label>
              <select class="u-input">
                <option>{{ t('settings.system.months3') }}</option>
                <option>{{ t('settings.system.months6') }}</option>
                <option>{{ t('settings.system.year1') }}</option>
                <option>{{ t('settings.system.years2') }}</option>
              </select>
            </div>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">{{ t('settings.system.autoBackup') }}</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">{{ t('settings.system.autoBackupDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" checked class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="pt-4">
            <button class="u-btn u-btn-primary u-btn-md">
              <Icon name="heroicons:check" class="w-4 h-4 mr-2" />
              {{ t('settings.system.saveSystemSettings') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  title: 'Settings',
  layout: 'dashboard',
  middleware: ['auth']
})

const { t } = useI18n()
const activeTab = ref('profile')

// FIXED: Make tabs reactive to locale changes
const tabs = computed(() => [
  { id: 'profile', name: t('settings.tabs.profile'), icon: 'heroicons:user' },
  { id: 'notifications', name: t('settings.tabs.notifications'), icon: 'heroicons:bell' },
  { id: 'integrations', name: t('settings.tabs.integrations'), icon: 'heroicons:puzzle-piece' },
  { id: 'team', name: t('settings.tabs.team'), icon: 'heroicons:users' },
  { id: 'system', name: t('settings.tabs.system'), icon: 'heroicons:cog-6-tooth' }
])

const profile = ref({ firstName: 'John', lastName: 'Doe', email: 'john.doe@company.com', company: 'ABC Manufacturing', phone: '+1 (555) 123-4567' })

const notifications = ref({ systemAlerts: true, maintenanceReminders: true, diagnosticReports: true, weeklySummary: false })
</script>
