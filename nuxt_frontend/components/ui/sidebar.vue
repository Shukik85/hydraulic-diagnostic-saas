<template>
  <div class="flex h-screen">
    <!-- Sidebar -->
    <aside
      :class="[
        'bg-card border-r border-border transition-all duration-300 ease-in-out flex flex-col',
        isCollapsed ? 'w-16' : 'w-70'
      ]"
    >
      <!-- Header -->
      <div class="p-4 border-b border-border flex items-center justify-between">
        <div
          :class="[
            'flex items-center gap-3 transition-opacity duration-200',
            isCollapsed ? 'opacity-0' : 'opacity-100'
          ]"
        >
          <div class="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
            <Icon name="lucide:gauge" class="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h2 class="font-semibold text-sm">HydraulicsTell</h2>
            <p class="text-xs text-muted-foreground">Диагностика</p>
          </div>
        </div>
        <UiButton
          variant="ghost"
          size="icon"
          @click="toggleSidebar"
          class="h-8 w-8 shrink-0"
        >
          <Icon
            :name="isCollapsed ? 'lucide:chevron-right' : 'lucide:chevron-left'"
            class="h-4 w-4"
          />
        </UiButton>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 p-2 space-y-1">
        <UiButton
          v-for="item in navigationItems"
          :key="item.key"
          :variant="activeItem === item.key ? 'secondary' : 'ghost'"
          :class="[
            'w-full justify-start gap-3 h-10 px-3 transition-all duration-200',
            isCollapsed ? 'px-2' : 'px-3'
          ]"
          @click="setActiveItem(item.key)"
        >
          <Icon :name="item.icon" class="h-4 w-4 shrink-0" />
          <span
            :class="[
              'transition-opacity duration-200 truncate',
              isCollapsed ? 'opacity-0 w-0' : 'opacity-100'
            ]"
          >
            {{ item.label }}
          </span>
        </UiButton>
      </nav>

      <!-- Footer -->
      <div class="p-2 border-t border-border">
        <UiButton
          variant="ghost"
          :class="[
            'w-full justify-start gap-3 h-10 px-3 transition-all duration-200',
            isCollapsed ? 'px-2' : 'px-3'
          ]"
          @click="handleLogout"
        >
          <Icon name="lucide:log-out" class="h-4 w-4 shrink-0" />
          <span
            :class="[
              'transition-opacity duration-200 truncate',
              isCollapsed ? 'opacity-0 w-0' : 'opacity-100'
            ]"
          >
            Выход
          </span>
        </UiButton>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 overflow-hidden">
      <slot />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface NavigationItem {
  key: string
  label: string
  icon: string
}

const props = defineProps<{
  defaultCollapsed?: boolean
}>()

const emit = defineEmits<{
  'item-click': [key: string]
  'logout': []
}>()

const isCollapsed = ref(props.defaultCollapsed || false)
const activeItem = ref('dashboard')

const navigationItems: NavigationItem[] = [
  { key: 'dashboard', label: 'Dashboard', icon: 'lucide:layout-dashboard' },
  { key: 'equipment', label: 'Equipment', icon: 'lucide:cog' },
  { key: 'sensors', label: 'Sensors', icon: 'lucide:thermometer' },
  { key: 'diagnostics', label: 'Diagnostics', icon: 'lucide:activity' },
  { key: 'chat', label: 'AI Chat', icon: 'lucide:message-square' },
  { key: 'reports', label: 'Reports', icon: 'lucide:file-text' },
  { key: 'settings', label: 'Settings', icon: 'lucide:settings' },
]

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

const setActiveItem = (key: string) => {
  activeItem.value = key
  emit('item-click', key)
}

const handleLogout = () => {
  emit('logout')
}

// Expose methods for parent components
defineExpose({
  toggle: toggleSidebar,
  collapse: () => isCollapsed.value = true,
  expand: () => isCollapsed.value = false,
})
</script>
