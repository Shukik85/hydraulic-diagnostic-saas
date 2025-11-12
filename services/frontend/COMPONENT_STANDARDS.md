# Component Usage Standards - Hydraulic Diagnostic SaaS

Этот документ определяет стандарты использования компонентов для обеспечения консистентности во всём проекте.

---

## 📌 Core Principle

> **ВСЕГДА используй Nuxt UI компоненты + Tailwind classes**

---

## 🔲 Buttons

### ✅ CORRECT

```vue
<!-- Primary action -->
<UButton color="primary" size="md" @click="action">
  Submit
</UButton>

<!-- Secondary action -->
<UButton color="gray" variant="outline" size="md">
  Cancel
</UButton>

<!-- Destructive action -->
<UButton color="red" size="md">
  Delete
</UButton>

<!-- With icon -->
<UButton color="primary" icon="i-heroicons-plus">
  Add Item
</UButton>

<!-- Icon only -->
<UButton
  color="gray"
  variant="ghost"
  icon="i-heroicons-cog-6-tooth"
  square
/>

<!-- Loading state -->
<UButton color="primary" :loading="isLoading">
  Save Changes
</UButton>
```

### ❌ INCORRECT

```vue
<!-- НЕ используй BaseButton -->
<BaseButton variant="primary">Submit</BaseButton>

<!-- НЕ используй button.vue -->
<Button>Submit</Button>

<!-- НЕ смешивай u-btn classes -->
<button class="u-btn u-btn-primary">Submit</button>
```

### Color Options

| Color | Usage |
|-------|-------|
| `primary` | Основное действие (Submit, Save, Create) |
| `gray` | Вторичное действие (Cancel, Back) |
| `red` | Destructive действие (Delete, Remove) |
| `green` | Success действие (Approve, Confirm) |
| `blue` | Info действие (Details, View) |

### Variant Options

| Variant | Usage |
|---------|-------|
| `solid` (default) | Основные кнопки |
| `outline` | Вторичные кнопки |
| `soft` | Софт background |
| `ghost` | Минимальные кнопки, icon buttons |
| `link` | Link-style buttons |

### Size Options

| Size | Usage |
|------|-------|
| `xs` | Мини кнопки (таблицы, тэги) |
| `sm` | Маленькие кнопки (cards, inline) |
| `md` (default) | Стандартные кнопки |
| `lg` | Большие кнопки (CTAs, modals) |
| `xl` | Hero CTAs |

---

## 📋 Cards

### ✅ CORRECT

```vue
<!-- Basic card -->
<UCard class="p-6">
  <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
    Card Title
  </h3>
  <p class="text-sm text-gray-600 dark:text-gray-400">
    Card content
  </p>
</UCard>

<!-- Card with header slot -->
<UCard>
  <template #header>
    <div class="flex items-center justify-between">
      <h3 class="text-lg font-semibold">Title</h3>
      <UBadge color="green">Active</UBadge>
    </div>
  </template>
  
  <div class="space-y-4">
    <p>Content</p>
  </div>
  
  <template #footer>
    <div class="flex justify-end gap-3">
      <UButton color="gray">Cancel</UButton>
      <UButton color="primary">Save</UButton>
    </div>
  </template>
</UCard>

<!-- Hoverable card -->
<UCard class="p-6 cursor-pointer hover:shadow-lg transition-shadow">
  <!-- Content -->
</UCard>
```

### ❌ INCORRECT

```vue
<!-- НЕ используй BaseCard -->
<BaseCard hover>
  <template #header>Title</template>
</BaseCard>

<!-- НЕ используй u-card class -->
<div class="u-card u-card-hover">
```

### Padding Standards

| Context | Padding |
|---------|----------|
| Content cards | `p-6` |
| Compact cards | `p-4` |
| Large cards | `p-8` |
| Card sections | `space-y-4` or `space-y-6` |

---

## 🏷️ Badges

### ✅ CORRECT

```vue
<!-- Status badge -->
<UBadge color="green" variant="soft">
  Operational
</UBadge>

<!-- With icon -->
<UBadge color="yellow" variant="soft">
  <UIcon name="i-heroicons-exclamation-triangle" class="w-3 h-3" />
  Warning
</UBadge>

<!-- Count badge -->
<div class="relative">
  <UButton icon="i-heroicons-bell" />
  <UBadge 
    color="red" 
    class="absolute -top-1 -right-1"
    size="xs"
  >
    {{ notificationCount }}
  </UBadge>
</div>
```

### ❌ INCORRECT

```vue
<!-- НЕ используй StatusBadge -->
<StatusBadge status="operational" />

<!-- НЕ используй custom badge classes -->
<span class="u-badge u-badge-success">Active</span>
```

### Color Mapping

| Status | UBadge Color |
|--------|-------------|
| Operational / Success | `green` |
| Warning / Degraded | `yellow` |
| Error / Critical | `red` |
| Info / Processing | `blue` |
| Unknown / Disabled | `gray` |

---

## 📝 Form Inputs

### ✅ CORRECT

```vue
<!-- Text input -->
<UFormGroup label="Equipment Name" name="name" required>
  <UInput
    v-model="form.name"
    placeholder="Enter equipment name"
    icon="i-heroicons-tag"
  />
</UFormGroup>

<!-- Select -->
<UFormGroup label="Equipment Type" name="type">
  <USelect
    v-model="form.type"
    :options="equipmentTypes"
    placeholder="Select type"
  />
</UFormGroup>

<!-- Textarea -->
<UFormGroup label="Description" name="description">
  <UTextarea
    v-model="form.description"
    :rows="4"
    placeholder="Enter description"
  />
</UFormGroup>

<!-- Checkbox -->
<UCheckbox v-model="form.isActive" label="Active" />

<!-- Radio group -->
<URadioGroup
  v-model="form.priority"
  :options="[
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' }
  ]"
/>
```

### ❌ INCORRECT

```vue
<!-- НЕ используй u-input classes -->
<input class="u-input" />

<!-- НЕ используй u-label -->
<label class="u-label u-label-required">Name</label>
```

---

## 💡 Modals & Dialogs

### ✅ CORRECT

```vue
<template>
  <UModal v-model="isOpen" :ui="{ width: 'sm:max-w-2xl' }">
    <UCard>
      <template #header>
        <div class="flex items-center justify-between">
          <h3 class="text-lg font-semibold">Modal Title</h3>
          <UButton
            color="gray"
            variant="ghost"
            icon="i-heroicons-x-mark"
            @click="isOpen = false"
          />
        </div>
      </template>
      
      <div class="space-y-4">
        <!-- Modal content -->
      </div>
      
      <template #footer>
        <div class="flex justify-end gap-3">
          <UButton color="gray" @click="isOpen = false">
            Cancel
          </UButton>
          <UButton color="primary" @click="submit">
            Submit
          </UButton>
        </div>
      </template>
    </UCard>
  </UModal>
</template>
```

### Modal Sizes

| Size | Width | Usage |
|------|-------|-------|
| `xs` | `sm:max-w-xs` | Подтверждения |
| `sm` | `sm:max-w-sm` | Простые формы |
| `md` | `sm:max-w-md` (default) | Стандартные формы |
| `lg` | `sm:max-w-2xl` | Сложные формы |
| `xl` | `sm:max-w-4xl` | Wizards, мульти-степ |

---

## 🚨 Alerts & Notifications

### ✅ CORRECT

```vue
<!-- Inline alert -->
<UAlert
  color="yellow"
  icon="i-heroicons-exclamation-triangle"
  title="Warning"
  description="This action cannot be undone"
/>

<!-- With actions -->
<UAlert color="red" title="Error occurred">
  <template #actions>
    <UButton size="xs" color="red" variant="outline">
      Retry
    </UButton>
  </template>
</UAlert>

<!-- Toast notification -->
const toast = useToast()

toast.add({
  title: 'Success',
  description: 'Operation completed',
  color: 'green',
  timeout: 3000
})
```

### Alert Colors

| Color | Usage |
|-------|-------|
| `green` | Success messages |
| `yellow` | Warnings |
| `red` | Errors |
| `blue` | Informational |
| `gray` | Neutral |

---

## 📏 Tables

### ✅ CORRECT

```vue
<UTable
  :rows="items"
  :columns="[
    { key: 'name', label: 'Name' },
    { key: 'status', label: 'Status' },
    { key: 'actions', label: 'Actions' }
  ]"
>
  <template #status-data="{ row }">
    <UBadge :color="getStatusColor(row.status)">
      {{ row.status }}
    </UBadge>
  </template>
  
  <template #actions-data="{ row }">
    <div class="flex gap-2">
      <UButton size="xs" @click="edit(row)">
        Edit
      </UButton>
      <UButton size="xs" color="red" @click="delete(row)">
        Delete
      </UButton>
    </div>
  </template>
</UTable>
```

### ❌ INCORRECT

```vue
<!-- НЕ используй u-table classes -->
<table class="u-table">
  <thead>
    <tr class="u-table-header">
```

---

## 📊 Charts

### ✅ CORRECT

```vue
<template>
  <UCard class="p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
        Chart Title
      </h3>
      <USelectMenu
        v-model="timeRange"
        :options="timeRangeOptions"
        size="sm"
      />
    </div>
    
    <div class="h-[300px] sm:h-[400px]">
      <v-chart
        :option="chartOption"
        autoresize
        class="w-full h-full"
      />
    </div>
  </UCard>
</template>
```

### Chart Heights

| Context | Height |
|---------|--------|
| Dashboard widgets | `h-[300px]` |
| Full-width charts | `h-[400px] lg:h-[500px]` |
| Detailed analysis | `h-[500px] lg:h-[600px]` |
| Mobile | Always use responsive |

---

## 🎨 Typography

### ✅ CORRECT

```vue
<!-- Headings -->
<h1 class="text-3xl font-bold text-gray-900 dark:text-gray-100">
<h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
<h3 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
<h4 class="text-lg font-semibold text-gray-900 dark:text-gray-100">

<!-- Body text -->
<p class="text-base text-gray-700 dark:text-gray-300">
<p class="text-sm text-gray-600 dark:text-gray-400">
<p class="text-xs text-gray-500 dark:text-gray-500">

<!-- Labels -->
<label class="text-sm font-medium text-gray-700 dark:text-gray-300">
```

### ❌ INCORRECT

```vue
<!-- НЕ используй u-h* classes -->
<h1 class="u-h1">
<h2 class="u-h2">

<!-- НЕ используй u-body -->
<p class="u-body u-body-lg">
```

---

## 🏛️ Layout & Spacing

### ✅ CORRECT

```vue
<!-- Page container -->
<div class="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
  <!-- Page content -->
</div>

<!-- Sections -->
<div class="space-y-8">         <!-- Large sections -->
  <section class="space-y-6">  <!-- Medium sections -->
    <div class="space-y-4">    <!-- Small sections -->
```

<!-- Grid layouts -->
<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">

<!-- Flex layouts -->
<div class="flex items-center gap-3">
<div class="flex justify-between gap-4">
```

### Spacing Scale

| Gap | Usage |
|-----|-------|
| `gap-2` (8px) | Icon + text, tight groups |
| `gap-3` (12px) | Button groups, inline elements |
| `gap-4` (16px) | Form fields, card items |
| `gap-6` (24px) | Cards grid, sections |
| `gap-8` (32px) | Large sections, page blocks |

---

## 🔸 Icons

### ✅ CORRECT

```vue
<!-- Nuxt Icon (UIcon) -->
<UIcon name="i-heroicons-plus" class="w-5 h-5" />

<!-- In buttons -->
<UButton icon="i-heroicons-arrow-right">
  Next
</UButton>

<!-- Sizes -->
<UIcon name="i-heroicons-home" class="w-4 h-4" />  <!-- Small -->
<UIcon name="i-heroicons-home" class="w-5 h-5" />  <!-- Medium -->
<UIcon name="i-heroicons-home" class="w-6 h-6" />  <!-- Large -->
<UIcon name="i-heroicons-home" class="w-8 h-8" />  <!-- XL -->
```

### ❌ INCORRECT

```vue
<!-- НЕ используй Icon без i- prefix -->
<Icon name="heroicons:plus" />

<!-- НЕ используй разные icon библиотеки -->
<Icon name="mdi:plus" />
```

### Icon Library

**Используй ТОЛЬКО Heroicons:**
- Prefix: `i-heroicons-`
- Style: outline (default) or solid (`-solid` suffix)
- Example: `i-heroicons-check-circle`, `i-heroicons-check-circle-solid`

---

## 🔄 Loading States

### ✅ CORRECT

```vue
<!-- Button loading -->
<UButton color="primary" :loading="isLoading">
  Submit
</UButton>

<!-- Page loading -->
<div v-if="isLoading" class="space-y-4">
  <USkeleton class="h-12 w-full" />
  <USkeleton class="h-32 w-full" />
  <USkeleton class="h-64 w-full" />
</div>

<!-- Inline loading -->
<div v-if="isLoading" class="flex items-center justify-center py-12">
  <UIcon name="i-heroicons-arrow-path" class="w-8 h-8 animate-spin text-blue-500" />
</div>
```

### ❌ INCORRECT

```vue
<!-- НЕ используй u-spinner -->
<div class="u-spinner w-8 h-8"></div>

<!-- НЕ используй u-skeleton -->
<div class="u-skeleton h-20" />
```

---

## 🌐 Responsive Design

### ✅ CORRECT

```vue
<!-- Mobile-first grid -->
<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 sm:gap-6">

<!-- Responsive padding -->
<div class="px-4 sm:px-6 lg:px-8 py-6 sm:py-8">

<!-- Responsive text -->
<h1 class="text-2xl sm:text-3xl lg:text-4xl font-bold">

<!-- Hide on mobile -->
<div class="hidden lg:block">

<!-- Show only on mobile -->
<div class="block lg:hidden">
```

### Breakpoints

| Breakpoint | Min Width | Usage |
|------------|-----------|-------|
| `sm` | 640px | Tablet portrait |
| `md` | 768px | Tablet landscape |
| `lg` | 1024px | Desktop |
| `xl` | 1280px | Large desktop |
| `2xl` | 1536px | Extra large |

---

## 🎨 Colors

### ✅ CORRECT

```vue
<!-- Backgrounds -->
<div class="bg-white dark:bg-gray-800">
<div class="bg-gray-50 dark:bg-gray-900">
<div class="bg-blue-50 dark:bg-blue-900/20">

<!-- Text -->
<p class="text-gray-900 dark:text-gray-100">  <!-- Headings -->
<p class="text-gray-700 dark:text-gray-300">  <!-- Body -->
<p class="text-gray-600 dark:text-gray-400">  <!-- Muted -->
<p class="text-gray-500 dark:text-gray-500">  <!-- Very muted -->

<!-- Borders -->
<div class="border border-gray-200 dark:border-gray-700">

<!-- Accent colors -->
<div class="text-blue-600 dark:text-blue-400">
<div class="text-green-600 dark:text-green-400">
<div class="text-red-600 dark:text-red-400">
```

### Color Usage Guidelines

| Context | Light Mode | Dark Mode |
|---------|------------|----------|
| Page background | `bg-white` or `bg-gray-50` | `bg-gray-900` or `bg-gray-950` |
| Card background | `bg-white` | `bg-gray-800` |
| Heading text | `text-gray-900` | `text-gray-100` |
| Body text | `text-gray-700` | `text-gray-300` |
| Muted text | `text-gray-600` | `text-gray-400` |
| Border | `border-gray-200` | `border-gray-700` |
| Accent (primary) | `text-blue-600` | `text-blue-400` |
| Success | `text-green-600` | `text-green-400` |
| Warning | `text-yellow-600` | `text-yellow-400` |
| Error | `text-red-600` | `text-red-400` |

---

## 📑 Example Components

### Equipment Card (Complete Example)

```vue
<template>
  <UCard class="p-6 hover:shadow-lg transition-shadow cursor-pointer">
    <!-- Header -->
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <UIcon 
          name="i-heroicons-cpu-chip" 
          class="w-10 h-10 text-blue-600 dark:text-blue-400" 
        />
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {{ equipment.name }}
          </h3>
          <p class="text-sm text-gray-500 dark:text-gray-400">
            {{ equipment.model }}
          </p>
        </div>
      </div>
      <UBadge :color="getStatusColor(equipment.status)" variant="soft">
        {{ equipment.status }}
      </UBadge>
    </div>
    
    <!-- Stats -->
    <div class="grid grid-cols-3 gap-4 mb-4">
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">Sensors</p>
        <p class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ equipment.sensorCount }}
        </p>
      </div>
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">Uptime</p>
        <p class="text-lg font-semibold text-green-600 dark:text-green-400">
          {{ equipment.uptime }}%
        </p>
      </div>
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">Alerts</p>
        <p class="text-lg font-semibold text-red-600 dark:text-red-400">
          {{ equipment.alertCount }}
        </p>
      </div>
    </div>
    
    <!-- Actions -->
    <div class="flex gap-2 pt-4 border-t border-gray-200 dark:border-gray-700">
      <UButton size="sm" color="primary" block>
        View Details
      </UButton>
      <UButton size="sm" color="gray" variant="outline">
        <UIcon name="i-heroicons-cog-6-tooth" class="w-4 h-4" />
      </UButton>
    </div>
  </UCard>
</template>

<script setup lang="ts">
interface Props {
  equipment: {
    name: string
    model: string
    status: string
    sensorCount: number
    uptime: number
    alertCount: number
  }
}

defineProps<Props>()

function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    operational: 'green',
    warning: 'yellow',
    critical: 'red',
    offline: 'gray'
  }
  return colors[status] || 'gray'
}
</script>
```

---

## ✅ Checklist для новых компонентов

Перед commit, проверь:

- [ ] Использую только `UButton`, `UCard`, `UBadge`, etc.
- [ ] Все цвета имеют dark mode variant
- [ ] Использую Tailwind spacing (gap-4, space-y-6, p-6)
- [ ] Typography через Tailwind classes
- [ ] Icons через `UIcon` с `i-heroicons-` prefix
- [ ] Loading states реализованы
- [ ] Error states реализованы
- [ ] Responsive design (mobile-first)
- [ ] TypeScript types определены
- [ ] Props задокументированы (JSDoc)

---

## 📚 Reference Links

- [Nuxt UI Documentation](https://ui.nuxt.com/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [Heroicons](https://heroicons.com/)
- Project Design Audit: `DESIGN_AUDIT.md`
- Phase 2 Documentation: `PHASE2_DIAGNOSTIC_VISUALIZATION.md`

---

**Last Updated:** 12 ноября 2025  
**Maintainer:** Frontend Team
