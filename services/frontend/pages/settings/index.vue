<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h1 class="text-3xl font-bold text-white">{{ t('settings.title') }}</h1>
      <p class="text-steel-shine mt-2">{{ t('settings.subtitle') }}</p>
    </div>

    <!-- Settings Navigation -->
    <div class="flex space-x-1 bg-steel-900/50 p-1 rounded-lg w-fit">
      <button 
        v-for="tab in tabs" 
        :key="tab.id" 
        @click="activeTab = tab.id"
        class="px-6 py-3 text-sm font-medium rounded-md transition-all"
        :class="activeTab === tab.id 
          ? 'bg-primary-600 text-white shadow-lg' 
          : 'text-steel-shine hover:text-white hover:bg-steel-800/50'"
      >
        <Icon :name="tab.icon" class="w-4 h-4 mr-2 inline" />
        {{ tab.name }}
      </button>
    </div>

    <!-- Profile Settings -->
    <div v-if="activeTab === 'profile'" class="space-y-6">
      <div class="card-glass p-6">
        <div class="border-b border-steel-700/50 pb-4 mb-6">
          <h3 class="text-xl font-bold text-white">{{ t('settings.profile.title') }}</h3>
          <p class="text-steel-shine mt-1">{{ t('settings.profile.subtitle') }}</p>
        </div>

        <div class="space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <UFormGroup
              :label="t('settings.profile.firstName')"
              helper="Ваше имя для отображения в системе"
            >
              <UInput v-model="profile.firstName" />
            </UFormGroup>
            <UFormGroup
              :label="t('settings.profile.lastName')"
              helper="Ваша фамилия для официальных документов"
            >
              <UInput v-model="profile.lastName" />
            </UFormGroup>
          </div>

          <UFormGroup
            :label="t('settings.profile.emailAddress')"
            helper="Используется для входа и уведомлений"
            required
          >
            <UInput v-model="profile.email" type="email" />
          </UFormGroup>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <UFormGroup
              :label="t('settings.profile.company')"
              helper="Название вашей организации"
            >
              <UInput v-model="profile.company" />
            </UFormGroup>
            <UFormGroup
              :label="t('settings.profile.phone')"
              helper="Контактный телефон"
            >
              <UInput v-model="profile.phone" type="tel" />
            </UFormGroup>
          </div>

          <div class="pt-4">
            <UButton size="lg" @click="saveProfile">
              <Icon name="heroicons:check" class="w-5 h-5 mr-2" />
              {{ t('settings.profile.saveChanges') }}
            </UButton>
          </div>
        </div>
      </div>

      <div class="card-glass p-6">
        <div class="border-b border-steel-700/50 pb-4 mb-6">
          <h3 class="text-xl font-bold text-white">{{ t('settings.profile.changePassword') }}</h3>
          <p class="text-steel-shine mt-1">{{ t('settings.profile.changePasswordSubtitle') }}</p>
        </div>

        <div class="space-y-4">
          <UFormGroup
            :label="t('settings.profile.currentPassword')"
            helper="Введите текущий пароль для подтверждения"
            required
          >
            <UInput 
              v-model="passwords.current"
              type="password" 
              :placeholder="t('settings.profile.currentPasswordPlaceholder')" 
            />
          </UFormGroup>
          
          <UFormGroup
            :label="t('settings.profile.newPassword')"
            helper="Минимум 8 символов, включая цифры и спецсимволы"
            required
          >
            <UInput 
              v-model="passwords.new"
              type="password" 
              :placeholder="t('settings.profile.newPasswordPlaceholder')" 
            />
          </UFormGroup>
          
          <UFormGroup
            :label="t('settings.profile.confirmPassword')"
            helper="Повторите новый пароль"
            required
          >
            <UInput 
              v-model="passwords.confirm"
              type="password" 
              :placeholder="t('settings.profile.confirmPasswordPlaceholder')" 
            />
          </UFormGroup>

          <div class="pt-4">
            <UButton size="lg" @click="updatePassword">
              <Icon name="heroicons:key" class="w-5 h-5 mr-2" />
              {{ t('settings.profile.updatePassword') }}
            </UButton>
          </div>
        </div>
      </div>
    </div>

    <!-- Notifications Settings -->
    <div v-if="activeTab === 'notifications'" class="space-y-6">
      <div class="card-glass p-6">
        <div class="border-b border-steel-700/50 pb-4 mb-6">
          <h3 class="text-xl font-bold text-white">{{ t('settings.notifications.title') }}</h3>
          <p class="text-steel-shine mt-1">{{ t('settings.notifications.subtitle') }}</p>
        </div>

        <div class="space-y-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-white">{{ t('settings.notifications.systemAlerts') }}</p>
              <p class="text-sm text-steel-shine">{{ t('settings.notifications.systemAlertsDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.systemAlerts" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-steel-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-500/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-steel-600 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>

          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-white">{{ t('settings.notifications.maintenanceReminders') }}</p>
              <p class="text-sm text-steel-shine">{{ t('settings.notifications.maintenanceRemindersDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.maintenanceReminders" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-steel-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-500/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-steel-600 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>

          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-white">{{ t('settings.notifications.diagnosticReports') }}</p>
              <p class="text-sm text-steel-shine">{{ t('settings.notifications.diagnosticReportsDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.diagnosticReports" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-steel-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-500/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-steel-600 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>

          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-white">{{ t('settings.notifications.weeklySummary') }}</p>
              <p class="text-sm text-steel-shine">{{ t('settings.notifications.weeklySummaryDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.weeklySummary" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-steel-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-500/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-steel-600 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>

          <div class="pt-4">
            <UButton size="lg" @click="saveNotifications">
              <Icon name="heroicons:check" class="w-5 h-5 mr-2" />
              Сохранить настройки уведомлений
            </UButton>
          </div>
        </div>
      </div>
    </div>

    <!-- Integrations Settings -->
    <div v-if="activeTab === 'integrations'" class="space-y-6">
      <div class="card-glass p-6">
        <div class="border-b border-steel-700/50 pb-4 mb-6">
          <h3 class="text-xl font-bold text-white">{{ t('settings.integrations.title') }}</h3>
          <p class="text-steel-shine mt-1">{{ t('settings.integrations.subtitle') }}</p>
        </div>

        <div class="space-y-4">
          <div class="flex items-center justify-between p-4 border border-steel-700/50 rounded-lg card-hover">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-blue-600/10 rounded-lg flex items-center justify-center">
                <Icon name="heroicons:cloud" class="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <p class="font-medium text-white">{{ t('settings.integrations.scadaSystem') }}</p>
                <p class="text-sm text-steel-shine">{{ t('settings.integrations.scadaSystemDesc') }}</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <UBadge variant="success">{{ t('settings.integrations.connected') }}</UBadge>
              <UButton variant="secondary" size="sm">{{ t('settings.integrations.configure') }}</UButton>
            </div>
          </div>

          <div class="flex items-center justify-between p-4 border border-steel-700/50 rounded-lg card-hover">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-green-600/10 rounded-lg flex items-center justify-center">
                <Icon name="heroicons:circle-stack" class="w-5 h-5 text-green-400" />
              </div>
              <div>
                <p class="font-medium text-white">{{ t('settings.integrations.erpSystem') }}</p>
                <p class="text-sm text-steel-shine">{{ t('settings.integrations.erpSystemDesc') }}</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <UBadge variant="warning">{{ t('settings.integrations.notConnected') }}</UBadge>
              <UButton size="sm">{{ t('settings.integrations.connect') }}</UButton>
            </div>
          </div>

          <div class="flex items-center justify-between p-4 border border-steel-700/50 rounded-lg card-hover">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-purple-600/10 rounded-lg flex items-center justify-center">
                <Icon name="heroicons:chat-bubble-left-right" class="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <p class="font-medium text-white">{{ t('settings.integrations.slackIntegration') }}</p>
                <p class="text-sm text-steel-shine">{{ t('settings.integrations.slackIntegrationDesc') }}</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <UBadge variant="warning">{{ t('settings.integrations.notConnected') }}</UBadge>
              <UButton size="sm">{{ t('settings.integrations.connect') }}</UButton>
            </div>
          </div>
        </div>
      </div>

      <div class="card-glass p-6">
        <div class="border-b border-steel-700/50 pb-4 mb-6">
          <h3 class="text-xl font-bold text-white">{{ t('settings.integrations.apiKeys') }}</h3>
          <p class="text-steel-shine mt-1">{{ t('settings.integrations.apiKeysSubtitle') }}</p>
        </div>

        <div class="space-y-4">
          <div class="flex items-center justify-between p-4 bg-steel-800/50 rounded-lg">
            <div>
              <p class="font-medium text-white">{{ t('settings.integrations.primaryApiKey') }}</p>
              <p class="text-sm text-steel-shine font-mono">hds_sk_...aBcD1234</p>
            </div>
            <div class="flex items-center gap-2">
              <UButton variant="secondary" size="sm">
                <Icon name="heroicons:clipboard" class="w-4 h-4 mr-1" />
                {{ t('settings.integrations.copy') }}
              </UButton>
              <UButton variant="ghost" size="icon">
                <Icon name="heroicons:arrow-path" class="w-5 h-5" />
              </UButton>
            </div>
          </div>

          <UButton variant="ghost" size="lg">
            <Icon name="heroicons:plus" class="w-5 h-5 mr-2" />
            {{ t('settings.integrations.generateNewKey') }}
          </UButton>
        </div>
      </div>
    </div>

    <!-- Team Settings -->
    <div v-if="activeTab === 'team'" class="space-y-6">
      <div class="flex items-center justify-between">
        <div>
          <h3 class="text-xl font-bold text-white">{{ t('settings.team.title') }}</h3>
          <p class="text-steel-shine mt-1">{{ t('settings.team.subtitle') }}</p>
        </div>
        <UButton size="lg">
          <Icon name="heroicons:user-plus" class="w-5 h-5 mr-2" />
          {{ t('settings.team.inviteUser') }}
        </UButton>
      </div>

      <div class="card-glass overflow-hidden">
        <div class="overflow-x-auto">
          <table class="w-full">
            <thead class="bg-steel-900/50 border-b border-steel-700/50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">{{ t('settings.team.user') }}</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">{{ t('settings.team.role') }}</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">{{ t('settings.team.lastActive') }}</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">{{ t('settings.team.status') }}</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">{{ t('settings.team.actions') }}</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-steel-700/50">
              <tr class="hover:bg-steel-900/30 transition-colors">
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center gap-3">
                    <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                      <Icon name="heroicons:user" class="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <p class="font-medium text-white">John Doe</p>
                      <p class="text-sm text-steel-shine">john.doe@company.com</p>
                    </div>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <UBadge variant="default">{{ t('settings.team.admin') }}</UBadge>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-steel-shine">{{ t('settings.team.hoursAgo', ['2']) }}</td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <UBadge variant="success">{{ t('settings.team.active') }}</UBadge>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center gap-1">
                    <UButton variant="ghost" size="icon">
                      <Icon name="heroicons:pencil" class="w-5 h-5" />
                    </UButton>
                    <UButton variant="ghost" size="icon">
                      <Icon name="heroicons:ellipsis-horizontal" class="w-5 h-5" />
                    </UButton>
                  </div>
                </td>
              </tr>

              <tr class="hover:bg-steel-900/30 transition-colors">
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center gap-3">
                    <div class="w-8 h-8 bg-gradient-to-br from-green-500 to-teal-500 rounded-full flex items-center justify-center">
                      <Icon name="heroicons:user" class="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <p class="font-medium text-white">Jane Smith</p>
                      <p class="text-sm text-steel-shine">jane.smith@company.com</p>
                    </div>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <UBadge variant="warning">{{ t('settings.team.engineer') }}</UBadge>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-steel-shine">{{ t('settings.team.dayAgo', ['1']) }}</td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <UBadge variant="success">{{ t('settings.team.active') }}</UBadge>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center gap-1">
                    <UButton variant="ghost" size="icon">
                      <Icon name="heroicons:pencil" class="w-5 h-5" />
                    </UButton>
                    <UButton variant="ghost" size="icon">
                      <Icon name="heroicons:ellipsis-horizontal" class="w-5 h-5" />
                    </UButton>
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
      <div class="card-glass p-6">
        <div class="border-b border-steel-700/50 pb-4 mb-6">
          <h3 class="text-xl font-bold text-white">{{ t('settings.system.title') }}</h3>
          <p class="text-steel-shine mt-1">{{ t('settings.system.subtitle') }}</p>
        </div>

        <div class="space-y-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <UFormGroup
              :label="t('settings.system.defaultAlertThreshold')"
              helper="Порог для автоматических предупреждений"
            >
              <USelect v-model="systemSettings.alertThreshold">
                <option value="75">{{ t('settings.system.warningAt75') }}</option>
                <option value="80">{{ t('settings.system.warningAt80') }}</option>
                <option value="90">{{ t('settings.system.warningAt90') }}</option>
              </USelect>
            </UFormGroup>
            
            <UFormGroup
              :label="t('settings.system.dataRetentionPeriod')"
              helper="Как долго хранить исторические данные"
            >
              <USelect v-model="systemSettings.dataRetention">
                <option value="3">{{ t('settings.system.months3') }}</option>
                <option value="6">{{ t('settings.system.months6') }}</option>
                <option value="12">{{ t('settings.system.year1') }}</option>
                <option value="24">{{ t('settings.system.years2') }}</option>
              </USelect>
            </UFormGroup>
          </div>

          <div class="flex items-center justify-between">
            <div>
              <p class="font-medium text-white">{{ t('settings.system.autoBackup') }}</p>
              <p class="text-sm text-steel-shine">{{ t('settings.system.autoBackupDesc') }}</p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="systemSettings.autoBackup" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-steel-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-500/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-steel-600 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>

          <div class="pt-4">
            <UButton size="lg" @click="saveSystemSettings">
              <Icon name="heroicons:check" class="w-5 h-5 mr-2" />
              {{ t('settings.system.saveSystemSettings') }}
            </UButton>
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

interface Tab {
  id: string
  name: string
  icon: string
}

const tabs = computed((): Tab[] => [
  { id: 'profile', name: t('settings.tabs.profile'), icon: 'heroicons:user' },
  { id: 'notifications', name: t('settings.tabs.notifications'), icon: 'heroicons:bell' },
  { id: 'integrations', name: t('settings.tabs.integrations'), icon: 'heroicons:puzzle-piece' },
  { id: 'team', name: t('settings.tabs.team'), icon: 'heroicons:users' },
  { id: 'system', name: t('settings.tabs.system'), icon: 'heroicons:cog-6-tooth' }
])

const profile = ref({ 
  firstName: 'John', 
  lastName: 'Doe', 
  email: 'john.doe@company.com', 
  company: 'ABC Manufacturing', 
  phone: '+1 (555) 123-4567' 
})

const passwords = ref({
  current: '',
  new: '',
  confirm: ''
})

const notifications = ref({ 
  systemAlerts: true, 
  maintenanceReminders: true, 
  diagnosticReports: true, 
  weeklySummary: false 
})

const systemSettings = ref({
  alertThreshold: '80',
  dataRetention: '12',
  autoBackup: true
})

const saveProfile = (): void => {
  console.log('Saving profile:', profile.value)
  // TODO: Implement API call
}

const updatePassword = (): void => {
  console.log('Updating password')
  // TODO: Implement API call
}

const saveNotifications = (): void => {
  console.log('Saving notifications:', notifications.value)
  // TODO: Implement API call
}

const saveSystemSettings = (): void => {
  console.log('Saving system settings:', systemSettings.value)
  // TODO: Implement API call
}
</script>
