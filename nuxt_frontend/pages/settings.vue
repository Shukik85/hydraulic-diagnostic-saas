<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h1 class="u-h2">Settings</h1>
      <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
        Manage your account and system preferences
      </p>
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
          <h3 class="u-h4">Personal Information</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Update your personal details and contact information
          </p>
        </div>
        
        <div class="space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="u-label">First Name</label>
              <input v-model="profile.firstName" class="u-input" />
            </div>
            <div>
              <label class="u-label">Last Name</label>
              <input v-model="profile.lastName" class="u-input" />
            </div>
          </div>
          
          <div>
            <label class="u-label">Email Address</label>
            <input v-model="profile.email" type="email" class="u-input" />
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="u-label">Company</label>
              <input v-model="profile.company" class="u-input" />
            </div>
            <div>
              <label class="u-label">Phone</label>
              <input v-model="profile.phone" type="tel" class="u-input" />
            </div>
          </div>
          
          <div class="pt-4">
            <button class="u-btn u-btn-primary u-btn-md">
              <Icon name="heroicons:check" class="w-4 h-4 mr-2" />
              Save Changes
            </button>
          </div>
        </div>
      </div>

      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">Change Password</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Update your account password for security
          </p>
        </div>
        
        <div class="space-y-4">
          <div>
            <label class="u-label">Current Password</label>
            <input type="password" class="u-input" placeholder="Enter current password" />
          </div>
          <div>
            <label class="u-label">New Password</label>
            <input type="password" class="u-input" placeholder="Enter new password" />
          </div>
          <div>
            <label class="u-label">Confirm New Password</label>
            <input type="password" class="u-input" placeholder="Confirm new password" />
          </div>
          
          <div class="pt-4">
            <button class="u-btn u-btn-primary u-btn-md">
              <Icon name="heroicons:key" class="w-4 h-4 mr-2" />
              Update Password
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Notifications Settings -->
    <div v-if="activeTab === 'notifications'" class="space-y-6">
      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">Email Notifications</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Configure when you receive email alerts
          </p>
        </div>
        
        <div class="space-y-6">
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">System Alerts</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">
                Critical system failures and emergencies
              </p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.systemAlerts" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">Maintenance Reminders</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">
                Upcoming service and maintenance tasks
              </p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.maintenanceReminders" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">Diagnostic Reports</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">
                When diagnostic reports are ready
              </p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input v-model="notifications.diagnosticReports" type="checkbox" class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">Weekly Summary</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">
                Weekly system health overview
              </p>
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
          <h3 class="u-h4">System Integrations</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Connect with external systems and services
          </p>
        </div>
        
        <div class="space-y-4">
          <div class="u-flex-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg u-flex-center">
                <Icon name="heroicons:cloud" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">SCADA System</p>
                <p class="u-body-sm text-gray-500 dark:text-gray-400">Real-time data integration</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span class="u-badge u-badge-success">Connected</span>
              <button class="u-btn u-btn-secondary u-btn-sm">Configure</button>
            </div>
          </div>

          <div class="u-flex-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg u-flex-center">
                <Icon name="heroicons:circle-stack" class="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">ERP System</p>
                <p class="u-body-sm text-gray-500 dark:text-gray-400">Maintenance and inventory sync</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span class="u-badge u-badge-warning">Not Connected</span>
              <button class="u-btn u-btn-primary u-btn-sm">Connect</button>
            </div>
          </div>

          <div class="u-flex-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg u-flex-center">
                <Icon name="heroicons:chat-bubble-left-right" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">Slack Integration</p>
                <p class="u-body-sm text-gray-500 dark:text-gray-400">Team notifications and alerts</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span class="u-badge u-badge-warning">Not Connected</span>
              <button class="u-btn u-btn-primary u-btn-sm">Connect</button>
            </div>
          </div>
        </div>
      </div>

      <div class="u-card p-6">
        <div class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-6">
          <h3 class="u-h4">API Keys</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Manage API access tokens for external integrations
          </p>
        </div>
        
        <div class="space-y-4">
          <div class="u-flex-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">Primary API Key</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400 font-mono">hds_sk_...aBcD1234</p>
            </div>
            <div class="flex items-center gap-2">
              <button class="u-btn u-btn-secondary u-btn-sm">
                <Icon name="heroicons:clipboard" class="w-4 h-4 mr-1" />
                Copy
              </button>
              <button class="u-btn u-btn-ghost u-btn-sm">
                <Icon name="heroicons:arrow-path" class="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <button class="u-btn u-btn-ghost u-btn-md">
            <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
            Generate New Key
          </button>
        </div>
      </div>
    </div>

    <!-- Team Settings -->
    <div v-if="activeTab === 'team'" class="space-y-6">
      <div class="u-flex-between">
        <div>
          <h3 class="u-h4">Team Members</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Manage user access and permissions
          </p>
        </div>
        <button class="u-btn u-btn-primary u-btn-md">
          <Icon name="heroicons:user-plus" class="w-4 h-4 mr-2" />
          Invite User
        </button>
      </div>

      <div class="u-card">
        <div class="overflow-x-auto">
          <table class="u-table">
            <thead>
              <tr>
                <th>User</th>
                <th>Role</th>
                <th>Last Active</th>
                <th>Status</th>
                <th>Actions</th>
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
                <td><span class="u-badge u-badge-info">Admin</span></td>
                <td class="u-body-sm text-gray-500 dark:text-gray-400">2 hours ago</td>
                <td><span class="u-badge u-badge-success">Active</span></td>
                <td>
                  <div class="flex items-center gap-1">
                    <button class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:pencil" class="w-4 h-4" />
                    </button>
                    <button class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:ellipsis-horizontal" class="w-4 h-4" />
                    </button>
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
                <td><span class="u-badge u-badge-warning">Engineer</span></td>
                <td class="u-body-sm text-gray-500 dark:text-gray-400">1 day ago</td>
                <td><span class="u-badge u-badge-success">Active</span></td>
                <td>
                  <div class="flex items-center gap-1">
                    <button class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:pencil" class="w-4 h-4" />
                    </button>
                    <button class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:ellipsis-horizontal" class="w-4 h-4" />
                    </button>
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
          <h3 class="u-h4">System Configuration</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Configure system-wide settings and preferences
          </p>
        </div>
        
        <div class="space-y-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="u-label">Default Alert Threshold</label>
              <select class="u-input">
                <option>Warning at 75%</option>
                <option>Warning at 80%</option>
                <option>Warning at 90%</option>
              </select>
            </div>
            <div>
              <label class="u-label">Data Retention Period</label>
              <select class="u-input">
                <option>3 months</option>
                <option>6 months</option>
                <option>1 year</option>
                <option>2 years</option>
              </select>
            </div>
          </div>
          
          <div class="u-flex-between">
            <div>
              <p class="font-medium text-gray-900 dark:text-white">Auto-Backup</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">
                Automatically backup diagnostic data daily
              </p>
            </div>
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" checked class="sr-only peer" />
              <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          
          <div class="pt-4">
            <button class="u-btn u-btn-primary u-btn-md">
              <Icon name="heroicons:check" class="w-4 h-4 mr-2" />
              Save System Settings
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
  layout: 'default',
  middleware: ['auth']
})

const activeTab = ref('profile')

const tabs = [
  { id: 'profile', name: 'Profile', icon: 'heroicons:user' },
  { id: 'notifications', name: 'Notifications', icon: 'heroicons:bell' },
  { id: 'integrations', name: 'Integrations', icon: 'heroicons:puzzle-piece' },
  { id: 'team', name: 'Team', icon: 'heroicons:users' },
  { id: 'system', name: 'System', icon: 'heroicons:cog-6-tooth' }
]

const profile = ref({
  firstName: 'John',
  lastName: 'Doe', 
  email: 'john.doe@company.com',
  company: 'ABC Manufacturing',
  phone: '+1 (555) 123-4567'
})

const notifications = ref({
  systemAlerts: true,
  maintenanceReminders: true,
  diagnosticReports: true,
  weeklySummary: false
})
</script>