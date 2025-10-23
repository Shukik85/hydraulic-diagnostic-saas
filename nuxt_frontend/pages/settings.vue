<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Settings</h1>
        <p class="text-muted-foreground">Manage your account and system preferences</p>
      </div>
    </div>

    <!-- Settings Tabs -->
    <UiTabs v-model="activeTab" class="w-full">
      <UiTabsList class="grid w-full grid-cols-5">
        <UiTabsTrigger value="profile">Profile</UiTabsTrigger>
        <UiTabsTrigger value="notifications">Notifications</UiTabsTrigger>
        <UiTabsTrigger value="integrations">Integrations</UiTabsTrigger>
        <UiTabsTrigger value="users">Users</UiTabsTrigger>
        <UiTabsTrigger value="billing">Billing</UiTabsTrigger>
      </UiTabsList>

      <!-- Profile Tab -->
      <UiTabsContent value="profile" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Personal Information</UiCardTitle>
            <UiCardDescription>Update your personal details and contact information</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <div class="space-y-2">
                <label class="text-sm font-medium">First Name</label>
                <UiInput v-model="profile.firstName" />
              </div>
              <div class="space-y-2">
                <label class="text-sm font-medium">Last Name</label>
                <UiInput v-model="profile.lastName" />
              </div>
            </div>
            <div class="space-y-2">
              <label class="text-sm font-medium">Email</label>
              <UiInput v-model="profile.email" type="email" />
            </div>
            <div class="space-y-2">
              <label class="text-sm font-medium">Company</label>
              <UiInput v-model="profile.company" />
            </div>
            <div class="space-y-2">
              <label class="text-sm font-medium">Phone</label>
              <UiInput v-model="profile.phone" type="tel" />
            </div>
            <UiButton>Save Changes</UiButton>
          </UiCardContent>
        </UiCard>

        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Change Password</UiCardTitle>
            <UiCardDescription>Update your account password</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="space-y-2">
              <label class="text-sm font-medium">Current Password</label>
              <UiInput type="password" />
            </div>
            <div class="space-y-2">
              <label class="text-sm font-medium">New Password</label>
              <UiInput type="password" />
            </div>
            <div class="space-y-2">
              <label class="text-sm font-medium">Confirm New Password</label>
              <UiInput type="password" />
            </div>
            <UiButton>Update Password</UiButton>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <!-- Notifications Tab -->
      <UiTabsContent value="notifications" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Email Notifications</UiCardTitle>
            <UiCardDescription>Configure when you receive email alerts</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">System Alerts</p>
                <p class="text-sm text-muted-foreground">Critical system failures and emergencies</p>
              </div>
              <UiCheckbox v-model="notifications.systemAlerts" />
            </div>
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">Maintenance Reminders</p>
                <p class="text-sm text-muted-foreground">Upcoming service and maintenance tasks</p>
              </div>
              <UiCheckbox v-model="notifications.maintenanceReminders" />
            </div>
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">Diagnostic Reports</p>
                <p class="text-sm text-muted-foreground">When diagnostic reports are ready</p>
              </div>
              <UiCheckbox v-model="notifications.diagnosticReports" />
            </div>
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">Weekly Summary</p>
                <p class="text-sm text-muted-foreground">Weekly system health overview</p>
              </div>
              <UiCheckbox v-model="notifications.weeklySummary" />
            </div>
          </UiCardContent>
        </UiCard>

        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Push Notifications</UiCardTitle>
            <UiCardDescription>Mobile and browser notifications</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">Browser Notifications</p>
                <p class="text-sm text-muted-foreground">Real-time alerts in your browser</p>
              </div>
              <UiCheckbox v-model="notifications.browserNotifications" />
            </div>
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">Mobile Alerts</p>
                <p class="text-sm text-muted-foreground">Push notifications to mobile devices</p>
              </div>
              <UiCheckbox v-model="notifications.mobileAlerts" />
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <!-- Integrations Tab -->
      <UiTabsContent value="integrations" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>API Integrations</UiCardTitle>
            <UiCardDescription>Connect with external systems and services</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="flex items-center justify-between p-4 border rounded-lg">
              <div class="flex items-center space-x-3">
                <div class="h-10 w-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Icon name="lucide:cloud" class="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p class="font-medium">SCADA System</p>
                  <p class="text-sm text-muted-foreground">Real-time data integration</p>
                </div>
              </div>
              <div class="flex items-center space-x-2">
                <span class="text-sm text-status-success">Connected</span>
                <UiButton variant="outline" size="sm">Configure</UiButton>
              </div>
            </div>

            <div class="flex items-center justify-between p-4 border rounded-lg">
              <div class="flex items-center space-x-3">
                <div class="h-10 w-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <Icon name="lucide:database" class="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p class="font-medium">ERP System</p>
                  <p class="text-sm text-muted-foreground">Maintenance and inventory sync</p>
                </div>
              </div>
              <div class="flex items-center space-x-2">
                <span class="text-sm text-muted-foreground">Not Connected</span>
                <UiButton variant="outline" size="sm">Connect</UiButton>
              </div>
            </div>

            <div class="flex items-center justify-between p-4 border rounded-lg">
              <div class="flex items-center space-x-3">
                <div class="h-10 w-10 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Icon name="lucide:slack" class="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p class="font-medium">Slack</p>
                  <p class="text-sm text-muted-foreground">Team notifications and alerts</p>
                </div>
              </div>
              <div class="flex items-center space-x-2">
                <span class="text-sm text-muted-foreground">Not Connected</span>
                <UiButton variant="outline" size="sm">Connect</UiButton>
              </div>
            </div>
          </UiCardContent>
        </UiCard>

        <UiCard>
          <UiCardHeader>
            <UiCardTitle>API Keys</UiCardTitle>
            <UiCardDescription>Manage API access tokens</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">Primary API Key</p>
                <p class="text-sm text-muted-foreground font-mono">sk-...aBcD</p>
              </div>
              <div class="flex items-center space-x-2">
                <UiButton variant="outline" size="sm">Regenerate</UiButton>
                <UiButton variant="outline" size="sm">Copy</UiButton>
              </div>
            </div>
            <UiButton variant="outline">
              <Icon name="lucide:plus" class="mr-2 h-4 w-4" />
              Generate New Key
            </UiButton>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <!-- Users Tab -->
      <UiTabsContent value="users" class="space-y-6">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-lg font-medium">Team Members</h3>
            <p class="text-sm text-muted-foreground">Manage user access and permissions</p>
          </div>
          <UiButton>
            <Icon name="lucide:user-plus" class="mr-2 h-4 w-4" />
            Invite User
          </UiButton>
        </div>

        <UiCard>
          <UiCardContent class="p-0">
            <div class="divide-y">
              <div class="flex items-center justify-between p-4">
                <div class="flex items-center space-x-3">
                  <div class="h-10 w-10 bg-primary/10 rounded-full flex items-center justify-center">
                    <Icon name="lucide:user" class="h-5 w-5" />
                  </div>
                  <div>
                    <p class="font-medium">John Doe</p>
                    <p class="text-sm text-muted-foreground">john.doe@company.com</p>
                  </div>
                </div>
                <div class="flex items-center space-x-2">
                  <span class="text-sm text-muted-foreground">Admin</span>
                  <UiButton variant="outline" size="sm">Edit</UiButton>
                </div>
              </div>

              <div class="flex items-center justify-between p-4">
                <div class="flex items-center space-x-3">
                  <div class="h-10 w-10 bg-primary/10 rounded-full flex items-center justify-center">
                    <Icon name="lucide:user" class="h-5 w-5" />
                  </div>
                  <div>
                    <p class="font-medium">Jane Smith</p>
                    <p class="text-sm text-muted-foreground">jane.smith@company.com</p>
                  </div>
                </div>
                <div class="flex items-center space-x-2">
                  <span class="text-sm text-muted-foreground">Engineer</span>
                  <UiButton variant="outline" size="sm">Edit</UiButton>
                </div>
              </div>

              <div class="flex items-center justify-between p-4">
                <div class="flex items-center space-x-3">
                  <div class="h-10 w-10 bg-primary/10 rounded-full flex items-center justify-center">
                    <Icon name="lucide:user" class="h-5 w-5" />
                  </div>
                  <div>
                    <p class="font-medium">Mike Johnson</p>
                    <p class="text-sm text-muted-foreground">mike.johnson@company.com</p>
                  </div>
                </div>
                <div class="flex items-center space-x-2">
                  <span class="text-sm text-muted-foreground">Viewer</span>
                  <UiButton variant="outline" size="sm">Edit</UiButton>
                </div>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <!-- Billing Tab -->
      <UiTabsContent value="billing" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Current Plan</UiCardTitle>
            <UiCardDescription>Your subscription and usage details</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <p class="font-medium">Enterprise Plan</p>
                <p class="text-sm text-muted-foreground">$299/month • 50 equipment units</p>
              </div>
              <UiButton variant="outline">Change Plan</UiButton>
            </div>
            <div class="space-y-2">
              <div class="flex justify-between text-sm">
                <span>Equipment Used</span>
                <span>24 / 50</span>
              </div>
              <div class="w-full bg-muted rounded-full h-2">
                <div class="bg-primary h-2 rounded-full" style="width: 48%"></div>
              </div>
            </div>
          </UiCardContent>
        </UiCard>

        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Payment Method</UiCardTitle>
            <UiCardDescription>Manage your billing information</UiCardDescription>
          </UiCardHeader>
          <UiCardContent class="space-y-4">
            <div class="flex items-center justify-between p-4 border rounded-lg">
              <div class="flex items-center space-x-3">
                <Icon name="lucide:credit-card" class="h-5 w-5" />
                <div>
                  <p class="font-medium">•••• •••• •••• 4242</p>
                  <p class="text-sm text-muted-foreground">Expires 12/25</p>
                </div>
              </div>
              <UiButton variant="outline" size="sm">Update</UiButton>
            </div>
          </UiCardContent>
        </UiCard>

        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Billing History</UiCardTitle>
            <UiCardDescription>View past invoices and payments</UiCardDescription>
          </UiCardHeader>
          <UiCardContent>
            <div class="space-y-3">
              <div class="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <p class="font-medium">January 2024</p>
                  <p class="text-sm text-muted-foreground">Invoice #INV-001</p>
                </div>
                <div class="flex items-center space-x-2">
                  <span class="text-sm font-medium">$299.00</span>
                  <UiButton variant="outline" size="sm">Download</UiButton>
                </div>
              </div>

              <div class="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <p class="font-medium">December 2023</p>
                  <p class="text-sm text-muted-foreground">Invoice #INV-002</p>
                </div>
                <div class="flex items-center space-x-2">
                  <span class="text-sm font-medium">$299.00</span>
                  <UiButton variant="outline" size="sm">Download</UiButton>
                </div>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>
    </UiTabs>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const activeTab = ref('profile')

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
  weeklySummary: false,
  browserNotifications: true,
  mobileAlerts: false
})
</script>
