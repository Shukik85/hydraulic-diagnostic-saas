<script setup lang="ts">
import { ref, computed } from 'vue'
import { useSeoMeta } from '#imports'

definePageMeta({
  title: 'Settings',
  layout: 'dashboard',
  middleware: ['auth']
})
const { t } = useI18n()
useSeoMeta({
  title: 'Настройки | Hydraulic Diagnostic SaaS',
  description: 'Профиль, безопасность, ключи API, интеграции и права доступа. Всё управление учётной записью в одном месте.',
  ogTitle: 'Settings | Hydraulic Diagnostic SaaS',
  ogDescription: 'Unified user profile, API keys, integrations and access for hydraulic diagnostics SaaS platform.',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})
const activeTab = ref('profile')
const profile = ref({ firstName: 'John', lastName: 'Doe', email: 'john.doe@company.com', company: 'ABC Manufacturing', phone: '+1 (555) 123-4567' })
const passwords = ref({ current: '', new: '', confirm: '' })
const notifications = ref({ systemAlerts: true, maintenanceReminders: true, diagnosticReports: true, weeklySummary: false })
const systemSettings = ref({ alertThreshold: '80', dataRetention: '12', autoBackup: true })
const saveProfile = (): void => {
  console.log('Saving profile:', profile.value)
}
const updatePassword = (): void => {
  console.log('Updating password')
}
const saveNotifications = (): void => {
  console.log('Saving notifications:', notifications.value)
}
const saveSystemSettings = (): void => {
  console.log('Saving system settings:', systemSettings.value)
}
</script>
