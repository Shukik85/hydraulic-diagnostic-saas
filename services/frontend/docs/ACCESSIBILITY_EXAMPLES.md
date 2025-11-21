# Accessibility Implementation Examples

Практические примеры применения A11y в Nuxt4 приложении.

---

## Оглавление

1. [Modal Dialog](#modal-dialog)
2. [Navigation Menu](#navigation-menu)
3. [Form with Validation](#form-with-validation)
4. [Loading States](#loading-states)
5. [Data Table](#data-table)
6. [Tabs Component](#tabs-component)
7. [Toast Notifications](#toast-notifications)

---

## Modal Dialog

### ❌ Before (No A11y)

```vue
<template>
  <div v-if="isOpen" class="modal">
    <div class="modal-content">
      <button @click="close">×</button>
      <h2>{{ title }}</h2>
      <slot />
    </div>
  </div>
</template>
```

### ✅ After (With A11y)

```vue
<script setup lang="ts">
import { ref, watch } from '#imports'
import { useAccessibility } from '~/composables/useAccessibility'

interface Props {
  isOpen: boolean
  title: string
}

const props = defineProps<Props>()
const emit = defineEmits<{
  close: []
}>()

const modalRef = ref<HTMLElement | null>(null)
const { activateFocusTrap, deactivateFocusTrap, restoreFocus } = useAccessibility()

watch(() => props.isOpen, (newValue) => {
  if (newValue) {
    activateFocusTrap(modalRef, {
      escapeDeactivates: true,
      onDeactivate: () => emit('close')
    })
  } else {
    deactivateFocusTrap()
    restoreFocus()
  }
})
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="isOpen" class="modal-container">
        <!-- Overlay -->
        <div 
          class="modal-overlay" 
          @click="emit('close')"
          aria-hidden="true"
        />
        
        <!-- Modal -->
        <div
          ref="modalRef"
          class="modal-content"
          role="dialog"
          aria-modal="true"
          :aria-labelledby="`modal-title-${$.uid}`"
        >
          <!-- Close button -->
          <button
            @click="emit('close')"
            class="modal-close"
            aria-label="Close dialog"
          >
            <Icon name="heroicons:x-mark" aria-hidden="true" />
          </button>
          
          <!-- Title -->
          <h2 :id="`modal-title-${$.uid}`" class="modal-title">
            {{ title }}
          </h2>
          
          <!-- Content -->
          <div class="modal-body">
            <slot />
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>
```

**Key improvements:**
- ✅ `role="dialog"` + `aria-modal="true"`
- ✅ `aria-labelledby` связывает с title
- ✅ Focus trap с Escape ключом
- ✅ Focus restoration после закрытия
- ✅ `aria-label` на close button

---

## Navigation Menu

### ❌ Before

```vue
<template>
  <div class="nav">
    <div 
      v-for="item in items" 
      :key="item.to"
      @click="navigate(item.to)"
    >
      {{ item.label }}
    </div>
  </div>
</template>
```

### ✅ After

```vue
<template>
  <nav aria-label="Main navigation">
    <ul role="list">
      <li v-for="item in items" :key="item.to">
        <NuxtLink
          :to="item.to"
          :aria-current="isActive(item.to) ? 'page' : undefined"
          :class="[
            'nav-link',
            { 'nav-link--active': isActive(item.to) }
          ]"
        >
          <Icon 
            :name="item.icon" 
            class="nav-icon"
            aria-hidden="true" 
          />
          <span>{{ item.label }}</span>
          <span 
            v-if="item.badge" 
            class="nav-badge"
            :aria-label="`${item.badge} new items`"
          >
            {{ item.badge }}
          </span>
        </NuxtLink>
      </li>
    </ul>
  </nav>
</template>
```

**Key improvements:**
- ✅ Semantic `<nav>` с `aria-label`
- ✅ Список `<ul>` + `<li>`
- ✅ `<NuxtLink>` вместо `<div @click>`
- ✅ `aria-current="page"` для active link
- ✅ `aria-hidden="true"` на decorative icons
- ✅ `aria-label` для badges

---

## Form with Validation

### ✅ Complete Example

```vue
<script setup lang="ts">
import { ref, computed } from '#imports'
import { useScreenReaderAnnounce } from '~/composables/useAccessibility'

const { announce } = useScreenReaderAnnounce()

const email = ref('')
const password = ref('')
const errors = ref<Record<string, string>>({})

const hasErrors = computed(() => Object.keys(errors.value).length > 0)

const validate = () => {
  errors.value = {}
  
  if (!email.value) {
    errors.value.email = 'Email is required'
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value)) {
    errors.value.email = 'Invalid email format'
  }
  
  if (!password.value) {
    errors.value.password = 'Password is required'
  } else if (password.value.length < 8) {
    errors.value.password = 'Password must be at least 8 characters'
  }
  
  return !hasErrors.value
}

const handleSubmit = async () => {
  if (!validate()) {
    announce('Form has errors. Please check the fields.', 'assertive')
    return
  }
  
  // Submit logic
  announce('Form submitted successfully', 'polite')
}
</script>

<template>
  <form @submit.prevent="handleSubmit" novalidate>
    <fieldset>
      <legend class="form-legend">Login Information</legend>
      
      <!-- Email Field -->
      <div class="form-group">
        <label for="email" class="form-label">
          Email Address
          <span aria-label="required">*</span>
        </label>
        <input
          id="email"
          v-model="email"
          type="email"
          class="form-input"
          required
          aria-required="true"
          aria-describedby="email-help email-error"
          :aria-invalid="!!errors.email"
          @blur="validate"
        />
        <p id="email-help" class="form-help">
          We'll never share your email
        </p>
        <p 
          v-if="errors.email" 
          id="email-error" 
          class="form-error"
          role="alert"
        >
          <Icon name="heroicons:exclamation-circle" aria-hidden="true" />
          {{ errors.email }}
        </p>
      </div>
      
      <!-- Password Field -->
      <div class="form-group">
        <label for="password" class="form-label">
          Password
          <span aria-label="required">*</span>
        </label>
        <input
          id="password"
          v-model="password"
          type="password"
          class="form-input"
          required
          aria-required="true"
          aria-describedby="password-help password-error"
          :aria-invalid="!!errors.password"
          @blur="validate"
        />
        <p id="password-help" class="form-help">
          Minimum 8 characters
        </p>
        <p 
          v-if="errors.password" 
          id="password-error" 
          class="form-error"
          role="alert"
        >
          <Icon name="heroicons:exclamation-circle" aria-hidden="true" />
          {{ errors.password }}
        </p>
      </div>
    </fieldset>
    
    <!-- Submit Button -->
    <button 
      type="submit" 
      class="btn btn-primary"
      :disabled="hasErrors"
    >
      Submit
    </button>
  </form>
</template>
```

**Key improvements:**
- ✅ `<label for="id">` для каждого input
- ✅ `aria-required="true"` + `required`
- ✅ `aria-describedby` связывает help text и errors
- ✅ `aria-invalid` для invalid inputs
- ✅ `role="alert"` для error messages
- ✅ Screen reader announcements

---

## Loading States

```vue
<template>
  <div 
    v-if="loading" 
    class="loading-container"
    role="status" 
    aria-live="polite"
  >
    <span class="sr-only">Loading data...</span>
    <Icon name="heroicons:arrow-path" class="animate-spin" aria-hidden="true" />
  </div>
  
  <div v-else>
    <!-- Content -->
  </div>
</template>
```

---

## Data Table

```vue
<template>
  <table role="table" aria-label="System diagnostics">
    <caption class="sr-only">System diagnostics table with sortable columns</caption>
    <thead>
      <tr>
        <th 
          scope="col"
          :aria-sort="sortBy === 'name' ? sortOrder : undefined"
          @click="sort('name')"
          class="sortable"
        >
          <button class="th-button">
            System Name
            <Icon 
              :name="getSortIcon('name')" 
              class="sort-icon"
              aria-hidden="true" 
            />
          </button>
        </th>
        <th scope="col">Status</th>
        <th scope="col">Last Check</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="system in systems" :key="system.id">
        <th scope="row">{{ system.name }}</th>
        <td>
          <span 
            :class="`status status--${system.status}`"
            :aria-label="`Status: ${system.status}`"
          >
            {{ system.status }}
          </span>
        </td>
        <td>
          <time :datetime="system.lastCheck">
            {{ formatDate(system.lastCheck) }}
          </time>
        </td>
      </tr>
    </tbody>
  </table>
</template>
```

---

## Tabs Component

```vue
<script setup lang="ts">
import { ref } from '#imports'

const activeTab = ref(0)

const tabs = [
  { id: 'overview', label: 'Overview' },
  { id: 'details', label: 'Details' },
  { id: 'history', label: 'History' }
]
</script>

<template>
  <div class="tabs">
    <!-- Tab List -->
    <div role="tablist" aria-label="System information">
      <button
        v-for="(tab, index) in tabs"
        :key="tab.id"
        :id="`tab-${tab.id}`"
        role="tab"
        :aria-selected="activeTab === index"
        :aria-controls="`panel-${tab.id}`"
        :tabindex="activeTab === index ? 0 : -1"
        :class="[
          'tab',
          { 'tab--active': activeTab === index }
        ]"
        @click="activeTab = index"
      >
        {{ tab.label }}
      </button>
    </div>
    
    <!-- Tab Panels -->
    <div
      v-for="(tab, index) in tabs"
      :key="`panel-${tab.id}`"
      :id="`panel-${tab.id}`"
      role="tabpanel"
      :aria-labelledby="`tab-${tab.id}`"
      :hidden="activeTab !== index"
      :tabindex="0"
      class="tab-panel"
    >
      <slot :name="tab.id" />
    </div>
  </div>
</template>
```

---

## Toast Notifications

```vue
<script setup lang="ts">
import { ref } from '#imports'
import { useScreenReaderAnnounce } from '~/composables/useAccessibility'

interface Toast {
  id: string
  message: string
  type: 'success' | 'error' | 'warning' | 'info'
}

const toasts = ref<Toast[]>([])
const { announce } = useScreenReaderAnnounce()

const addToast = (message: string, type: Toast['type'] = 'info') => {
  const id = Date.now().toString()
  toasts.value.push({ id, message, type })
  
  // Announce to screen readers
  announce(message, type === 'error' ? 'assertive' : 'polite')
  
  // Auto remove after 5s
  setTimeout(() => removeToast(id), 5000)
}

const removeToast = (id: string) => {
  toasts.value = toasts.value.filter(t => t.id !== id)
}
</script>

<template>
  <div 
    class="toast-container" 
    role="region" 
    aria-label="Notifications"
  >
    <TransitionGroup name="toast">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        :class="`toast toast--${toast.type}`"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <Icon 
          :name="getIcon(toast.type)" 
          class="toast-icon"
          aria-hidden="true" 
        />
        <p class="toast-message">{{ toast.message }}</p>
        <button
          @click="removeToast(toast.id)"
          class="toast-close"
          :aria-label="`Dismiss ${toast.type} notification`"
        >
          <Icon name="heroicons:x-mark" aria-hidden="true" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>
```

---

## Testing Checklist

### Manual Testing

- [ ] **Keyboard Navigation**
  - [ ] Tab через все элементы
  - [ ] Shift+Tab назад
  - [ ] Enter/Space для активации
  - [ ] Escape для закрытия
  - [ ] Arrow keys для navigation

- [ ] **Screen Reader** (NVDA/JAWS/VoiceOver)
  - [ ] Landmarks объявляются
  - [ ] Form labels читаются
  - [ ] Errors анонсируются
  - [ ] Live regions работают

- [ ] **Visual**
  - [ ] Focus visible на всех элементах
  - [ ] Color contrast ≥4.5:1
  - [ ] Text resizable до 200%

### Automated Testing

```bash
# Lighthouse
npm run build
lighthouse http://localhost:3000 --view

# axe DevTools
# Использовать browser extension
```

---

**Last Updated:** 19 ноября 2025
