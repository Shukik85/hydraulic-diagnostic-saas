# 🎨 Metallic Industrial Theme - Frontend Guide

**Version:** 1.0  
**Date:** November 13, 2025  
**Status:** ✅ Production Ready

---

## 📋 Overview

Металлический промышленный дизайн для Hydraulic Diagnostic SaaS Frontend.

### Ключевые особенности:
- ✅ **Brushed metal texture** - шлифованный металл
- ✅ **Inset shadows** - вдавленные тени
- ✅ **Industrial gradients** - промышленные градиенты
- ✅ **Muted indigo** - приглушенный индиго вместо яркого синего
- ✅ **Responsive** - адаптивный дизайн

---

## 🚀 Quick Start

### Metallic Card

```vue
<template>
  <div class="card-metal">
    <h3 class="text-xl font-bold mb-2">Excavator CAT-320D</h3>
    <p class="text-text-secondary">Status: Online</p>
    <div class="mt-4">
      <span class="badge-status badge-success">● Active</span>
    </div>
  </div>
</template>
```

### Buttons

```vue
<template>
  <div class="flex gap-4">
    <!-- Primary action -->
    <button class="btn-metal btn-primary">
      ✓ Save Changes
    </button>
    
    <!-- Secondary action -->
    <button class="btn-metal">
      Cancel
    </button>
  </div>
</template>
```

### Form Input

```vue
<template>
  <div>
    <label class="block text-text-secondary font-semibold mb-2 text-sm uppercase">
      System Name
    </label>
    <input 
      v-model="systemName"
      type="text" 
      class="input-metal"
      placeholder="Enter system name..."
    />
  </div>
</template>
```

---

## 🎨 Components

### 1. Metallic Cards (`.card-metal`)

**Usage:**
```vue
<div class="card-metal">
  <!-- Your content -->
</div>
```

**Features:**
- Brushed metal gradient background
- Inset shadows for depth
- Steel texture overlay
- Hover effect (lift + glow)

**Variants:**
```vue
<!-- With Tailwind utilities -->
<div class="card-metal p-6 hover:scale-105 transition">
  <h3 class="text-2xl font-bold">Title</h3>
</div>
```

### 2. Buttons (`.btn-metal`, `.btn-primary`)

**Primary Button:**
```vue
<button class="btn-metal btn-primary">
  Create System
</button>
```

**Secondary Button:**
```vue
<button class="btn-metal">
  Cancel
</button>
```

**With icon:**
```vue
<button class="btn-metal btn-primary">
  <span class="mr-2">✓</span>
  Save
</button>
```

### 3. Inputs (`.input-metal`)

**Text Input:**
```vue
<input type="text" class="input-metal" placeholder="Enter text..." />
```

**Select:**
```vue
<select class="input-metal">
  <option>Option 1</option>
  <option>Option 2</option>
</select>
```

**Textarea:**
```vue
<textarea class="input-metal" rows="4" placeholder="Description..."></textarea>
```

### 4. Status Badges

```vue
<template>
  <div class="flex gap-2">
    <span class="badge-status badge-success">● Online</span>
    <span class="badge-status badge-warning">⚠ Maintenance</span>
    <span class="badge-status badge-error">✗ Offline</span>
    <span class="badge-status badge-info">ℹ Info</span>
  </div>
</template>
```

---

## 🎯 Tailwind Utilities

### Colors

```vue
<!-- Metallic backgrounds -->
<div class="bg-metal-dark">Dark metal</div>
<div class="bg-metal-medium">Medium metal</div>
<div class="bg-metal-light">Light metal</div>

<!-- Steel accents -->
<div class="bg-steel-dark">Steel dark</div>
<div class="text-steel-shine">Steel shine text</div>

<!-- Primary (Indigo) -->
<div class="bg-primary-500">Primary background</div>
<div class="text-primary-300">Primary text</div>

<!-- Status colors -->
<div class="text-status-success">Success</div>
<div class="text-status-warning">Warning</div>
<div class="text-status-error">Error</div>
<div class="text-status-info">Info</div>

<!-- Background tokens -->
<div class="bg-background-primary">Primary bg</div>
<div class="bg-background-secondary">Secondary bg</div>
<div class="bg-background-tertiary">Tertiary bg</div>
```

### Gradients

```vue
<div class="bg-gradient-metal">Metal gradient</div>
<div class="bg-gradient-steel">Steel gradient</div>
<div class="bg-gradient-primary">Primary gradient</div>
<div class="bg-gradient-header">Header gradient</div>
```

### Shadows

```vue
<div class="shadow-metal">Metallic shadow</div>
<div class="shadow-inset-metal">Inset metal</div>
<div class="shadow-md">Medium shadow</div>
```

### Animations

```vue
<div class="animate-fade-in">Fade in</div>
<div class="animate-slide-in">Slide in</div>
<div class="animate-scale-in">Scale in</div>
<div class="animate-shine">Shine effect</div>
```

---

## 📝 Complete Examples

### Dashboard Card

```vue
<template>
  <div class="card-metal">
    <div class="flex justify-between items-start mb-4">
      <div>
        <h3 class="text-2xl font-bold">System Overview</h3>
        <p class="text-text-secondary text-sm">Last updated: 2 minutes ago</p>
      </div>
      <span class="badge-status badge-success">● Online</span>
    </div>
    
    <div class="grid grid-cols-2 gap-4">
      <div>
        <div class="text-text-muted text-sm uppercase">Total Systems</div>
        <div class="text-3xl font-bold text-primary-300">245</div>
      </div>
      <div>
        <div class="text-text-muted text-sm uppercase">Active</div>
        <div class="text-3xl font-bold text-status-success-light">128</div>
      </div>
    </div>
    
    <button class="btn-metal btn-primary w-full mt-4">
      View Details
    </button>
  </div>
</template>
```

### Form Example

```vue
<template>
  <div class="card-metal max-w-2xl mx-auto">
    <h2 class="text-2xl font-bold mb-6">Create New System</h2>
    
    <form @submit.prevent="handleSubmit">
      <div class="mb-4">
        <label class="block text-text-secondary font-semibold mb-2 text-sm uppercase">
          System Name *
        </label>
        <input 
          v-model="form.name"
          type="text" 
          class="input-metal"
          placeholder="Enter system name..."
          required
        />
      </div>
      
      <div class="mb-4">
        <label class="block text-text-secondary font-semibold mb-2 text-sm uppercase">
          Equipment Type *
        </label>
        <select v-model="form.type" class="input-metal" required>
          <option value="">Select type...</option>
          <option value="excavator">Excavator</option>
          <option value="loader">Loader</option>
          <option value="crane">Crane</option>
        </select>
      </div>
      
      <div class="mb-6">
        <label class="block text-text-secondary font-semibold mb-2 text-sm uppercase">
          Description
        </label>
        <textarea 
          v-model="form.description"
          class="input-metal"
          rows="4"
          placeholder="Enter description..."
        ></textarea>
      </div>
      
      <div class="flex gap-4">
        <button type="submit" class="btn-metal btn-primary flex-1">
          ✓ Create System
        </button>
        <button type="button" class="btn-metal" @click="cancel">
          Cancel
        </button>
      </div>
    </form>
  </div>
</template>

<script setup lang="ts">
const form = reactive({
  name: '',
  type: '',
  description: ''
})

const handleSubmit = () => {
  console.log('Creating system:', form)
}

const cancel = () => {
  navigateTo('/systems')
}
</script>
```

### Header with Shine Effect

```vue
<template>
  <header class="bg-gradient-header border-b border-primary-700 shadow-md header-shine">
    <div class="container mx-auto px-6 py-4 flex justify-between items-center">
      <h1 class="text-2xl font-bold flex items-center gap-2">
        <span>⚙️</span>
        Hydraulic Diagnostic SaaS
      </h1>
      
      <nav class="flex gap-6">
        <NuxtLink 
          to="/dashboard" 
          class="text-text-secondary hover:text-text-primary transition"
        >
          Dashboard
        </NuxtLink>
        <NuxtLink 
          to="/systems" 
          class="text-text-secondary hover:text-text-primary transition"
        >
          Systems
        </NuxtLink>
        <NuxtLink 
          to="/reports" 
          class="text-text-secondary hover:text-text-primary transition"
        >
          Reports
        </NuxtLink>
      </nav>
    </div>
  </header>
</template>
```

---

## 🔄 Migration Guide

### From Old Styling to Metallic

**Before:**
```vue
<div class="bg-gray-800 border border-gray-700 rounded-lg p-6">
  <button class="bg-primary-600 hover:bg-primary-700 px-4 py-2 rounded">
    Save
  </button>
</div>
```

**After:**
```vue
<div class="card-metal">
  <button class="btn-metal btn-primary">
    Save
  </button>
</div>
```

### Color Migration

```vue
<!-- OLD: Bright blue -->
<div class="bg-primary-500">  <!-- #3b82f6 -->

<!-- NEW: Muted indigo -->
<div class="bg-primary-500">  <!-- #4f46e5 -->
```

### Background Migration

```vue
<!-- OLD -->
<div class="bg-gray-900">        <!-- Generic gray -->
<div class="bg-gray-800">        <!-- Generic gray -->

<!-- NEW -->
<div class="bg-background-primary">   <!-- #0f1115 - Industrial dark -->
<div class="bg-background-secondary"> <!-- #2d3139 - Metallic -->
```

---

## ⚙️ Customization

### Changing Primary Color

In `tailwind.config.ts`:

```typescript
primary: {
  500: '#your-color',  // Main color
  600: '#darker-variant',
  700: '#darkest-variant',
}
```

### Custom Metal Shades

In `styles/metallic.css`:

```css
:root {
  --metal-dark: #your-dark-shade;
  --metal-medium: #your-medium-shade;
  --metal-light: #your-light-shade;
}
```

---

## 🎨 Tips & Best Practices

### 1. Use Semantic Classes

```vue
<!-- ✅ Good -->
<div class="bg-background-primary text-text-primary">

<!-- ❌ Avoid -->
<div class="bg-gray-900 text-gray-50">
```

### 2. Combine with Tailwind

```vue
<!-- ✅ Metallic base + Tailwind utilities -->
<div class="card-metal hover:scale-105 transition-transform">
```

### 3. Consistent Status Colors

```vue
<!-- ✅ Use badge classes -->
<span class="badge-status badge-success">● Online</span>

<!-- ❌ Avoid custom colors -->
<span class="bg-green-500 text-white">Online</span>
```

### 4. Accessibility

```vue
<!-- Always provide sufficient contrast -->
<button class="btn-metal" aria-label="Save changes">
  Save
</button>
```

---

## 🐛 Troubleshooting

### Styles not applying?

1. Check that `metallic.css` is imported in `app.vue`
2. Run `npm run dev` to rebuild
3. Clear browser cache

### Colors look different?

1. Ensure dark mode is enabled: `class="dark"`
2. Check CSS variables in browser DevTools

### Gradients not showing?

1. Verify Tailwind config includes `backgroundImage`
2. Use `bg-gradient-metal` not `bg-metal-gradient`

---

## 📚 Resources

- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Nuxt 4 Documentation](https://nuxt.com/docs)
- Design Tokens: `design-tokens.ts`
- CSS Variables: `styles/metallic.css`

---

**🎉 Готово! Metallic Industrial Theme применен к frontend!**