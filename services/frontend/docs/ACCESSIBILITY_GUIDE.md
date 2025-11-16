# Accessibility (A11y) Guide

## Общие принципы

Данное руководство основано на WCAG 2.1 Level AA и обеспечивает доступность приложения для всех пользователей.

---

## 1. Семантический HTML

### Правильные теги

```vue
<!-- ❌ ПЛОХО -->
<template>
  <div class="header">
    <div class="nav">
      <div class="link">Home</div>
      <div class="link">About</div>
    </div>
  </div>
</template>

<!-- ✅ ХОРОШО -->
<template>
  <header>
    <nav aria-label="Main navigation">
      <ul>
        <li>
          <NuxtLink to="/">Home</NuxtLink>
        </li>
        <li>
          <NuxtLink to="/about">About</NuxtLink>
        </li>
      </ul>
    </nav>
  </header>
</template>
```

### Главные ландмарки

```vue
<template>
  <div>
    <!-- Skip to main content link -->
    <a href="#main-content" class="sr-only-focusable">
      Skip to main content
    </a>
    
    <header>
      <nav aria-label="Primary navigation">
        <!-- Navigation -->
      </nav>
    </header>
    
    <main id="main-content">
      <!-- Main content -->
    </main>
    
    <aside aria-label="Sidebar">
      <!-- Sidebar content -->
    </aside>
    
    <footer>
      <!-- Footer content -->
    </footer>
  </div>
</template>

<style scoped>
/* Screen reader only */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

/* Visible only on focus */
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  overflow: visible;
  clip: auto;
  white-space: normal;
}
</style>
```

---

## 2. ARIA Атрибуты

### Кнопки и элементы управления

```vue
<template>
  <!-- Меню toggle -->
  <button
    @click="toggleMenu"
    :aria-expanded="isMenuOpen"
    aria-controls="mobile-menu"
    aria-label="Toggle navigation menu"
  >
    <Icon name="heroicons:bars-3" aria-hidden="true" />
  </button>
  
  <div
    id="mobile-menu"
    :aria-hidden="!isMenuOpen"
  >
    <!-- Menu content -->
  </div>
  
  <!-- Модальное окно -->
  <div
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
    aria-describedby="modal-description"
  >
    <h2 id="modal-title">Dialog Title</h2>
    <p id="modal-description">Dialog description</p>
  </div>
</template>
```

### Формы

```vue
<template>
  <form @submit.prevent="handleSubmit">
    <fieldset>
      <legend>Contact Information</legend>
      
      <!-- Input с полным accessibility -->
      <div>
        <label for="email">Email Address*</label>
        <input
          id="email"
          v-model="email"
          type="email"
          required
          aria-required="true"
          aria-describedby="email-help email-error"
          :aria-invalid="!!emailError"
        />
        <span id="email-help" class="help-text">
          We'll never share your email
        </span>
        <span 
          v-if="emailError" 
          id="email-error" 
          class="error-text"
          role="alert"
        >
          {{ emailError }}
        </span>
      </div>
      
      <!-- Select с ARIA -->
      <div>
        <label for="system-type">System Type*</label>
        <select
          id="system-type"
          v-model="systemType"
          required
          aria-required="true"
          aria-describedby="system-type-help"
        >
          <option value="">Select type...</option>
          <option value="hydraulic">Hydraulic</option>
          <option value="pneumatic">Pneumatic</option>
        </select>
        <span id="system-type-help" class="help-text">
          Choose your system type
        </span>
      </div>
    </fieldset>
    
    <button type="submit" :disabled="loading">
      <span v-if="loading" aria-live="polite">
        Submitting...
      </span>
      <span v-else>Submit</span>
    </button>
  </form>
</template>
```

### Live Regions

```vue
<template>
  <!-- Обновления в реальном времени -->
  <div 
    aria-live="polite" 
    aria-atomic="true"
    class="sr-only"
  >
    {{ statusMessage }}
  </div>
  
  <!-- Критические обновления -->
  <div 
    role="alert" 
    aria-live="assertive"
  >
    {{ errorMessage }}
  </div>
  
  <!-- Loading индикатор -->
  <div 
    v-if="loading"
    role="status" 
    aria-live="polite"
  >
    <span class="sr-only">Loading products...</span>
    <LoadingSpinner aria-hidden="true" />
  </div>
</template>
```

---

## 3. Keyboard Navigation

### Focus Management

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue'

const modalRef = ref<HTMLElement | null>(null)
const previousActiveElement = ref<HTMLElement | null>(null)

const openModal = () => {
  // Сохраняем текущий фокус
  previousActiveElement.value = document.activeElement as HTMLElement
  
  // Перемещаем фокус в модальное окно
  nextTick(() => {
    const firstFocusable = modalRef.value?.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as HTMLElement
    firstFocusable?.focus()
  })
}

const closeModal = () => {
  // Возвращаем фокус
  previousActiveElement.value?.focus()
}

// Focus trap
const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    closeModal()
    return
  }
  
  if (event.key === 'Tab') {
    const focusableElements = modalRef.value?.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
    
    if (!focusableElements?.length) return
    
    const first = focusableElements[0] as HTMLElement
    const last = focusableElements[focusableElements.length - 1] as HTMLElement
    
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div 
    ref="modalRef"
    role="dialog" 
    aria-modal="true"
  >
    <!-- Modal content -->
  </div>
</template>
```

### Focus Styles

```css
/* Global focus styles */
*:focus-visible {
  outline: 2px solid #21808D;
  outline-offset: 2px;
  border-radius: 2px;
}

button:focus-visible,
a:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible {
  outline: 2px solid #21808D;
  outline-offset: 2px;
}

/* Не убирайте outline полностью! */
/* ❌ ПЛОХО */
* {
  outline: none;
}

/* ✅ ХОРОШО - используйте :focus-visible */
```

---

## 4. Контраст Цветов

### WCAG 2.1 AA Требования

- **Обычный текст:** минимум 4.5:1
- **Крупный текст (18px+ или bold 14px+):** минимум 3:1
- **UI элементы:** минимум 3:1

### Проверка цветов

```css
/* Проверьте контраст на https://webaim.org/resources/contrastchecker/ */

:root {
  /* ✅ Текст на белом фоне */
  --color-text-primary: #1a1a1a;     /* 19.56:1 ✓ */
  --color-text-secondary: #4a4a4a;   /* 9.48:1 ✓ */
  --color-text-muted: #6b6b6b;       /* 5.74:1 ✓ */
  
  /* ✅ Brand цвета */
  --color-brand-primary: #21808D;    /* 4.52:1 ✓ на белом */
  --color-brand-hover: #1a6575;      /* 5.73:1 ✓ на белом */
  
  /* ✅ Status цвета */
  --color-success: #047857;          /* 4.76:1 ✓ */
  --color-warning: #b45309;          /* 4.65:1 ✓ */
  --color-error: #dc2626;            /* 4.53:1 ✓ */
  
  /* ❌ ПЛОХО - недостаточный контраст */
  /* --color-text-muted-bad: #c0c0c0; */ /* 2.98:1 ✗ */
}
```

---

## 5. Изображения и иконки

### Alt текст

```vue
<template>
  <!-- Декоративное изображение -->
  <img src="/decorative.jpg" alt="" aria-hidden="true" />
  
  <!-- Информативное изображение -->
  <img 
    src="/hydraulic-system.jpg" 
    alt="Hydraulic system diagnostic dashboard showing pressure levels"
  />
  
  <!-- Иконки -->
  <button>
    <Icon name="heroicons:trash" aria-hidden="true" />
    <span>Delete</span> <!-- Видимый текст -->
  </button>
  
  <!-- Иконка без текста -->
  <button aria-label="Delete system">
    <Icon name="heroicons:trash" aria-hidden="true" />
  </button>
</template>
```

---

## 6. Тестирование

### Инструменты

1. **Lighthouse** - встроенный в Chrome DevTools
2. **axe DevTools** - браузерное расширение
3. **NVDA / JAWS** - screen readers
4. **VoiceOver** - для macOS/iOS

### Чеклист тестирования

- [ ] Все элементы доступны с клавиатуры
- [ ] Focus visible на всех интерактивных элементах
- [ ] Контраст цветов ≥ 4.5:1
- [ ] Все изображения имеют alt текст
- [ ] Формы имеют labels
- [ ] ARIA атрибуты используются правильно
- [ ] Screen reader читает контент правильно
- [ ] Lighthouse accessibility score > 90

---

## 7. Дополнительные ресурсы

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Vue.js Accessibility Guide](https://vuejs.org/guide/best-practices/accessibility)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [axe DevTools](https://www.deque.com/axe/devtools/)

---

**Статус:** 🟢 Готов к применению  
**Последнее обновление:** 16 ноября 2025
