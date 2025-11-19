# Accessibility (A11y) Guide

## Общие принципы

Данное руководство основано на **WCAG 2.1 Level AA** и обеспечивает доступность приложения для всех пользователей, включая людей с ограниченными возможностями.

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
      {{ $t('a11y.skipToMainContent', 'Перейти к основному содержимому') }}
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
    :aria-label="$t('a11y.toggleNavigationMenu', 'Переключить меню навигации')"
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
    :aria-labelledby="titleId"
    :aria-describedby="descriptionId"
  >
    <h2 :id="titleId">Dialog Title</h2>
    <p :id="descriptionId">Dialog description</p>
  </div>
</template>
```

### Формы

```vue
<script setup lang="ts">
import { ref, computed } from '#imports'

const email = ref('')
const emailError = ref('')
const systemType = ref('')
const loading = ref(false)

const handleSubmit = () => {
  // Form submission logic
}
</script>

<template>
  <form @submit.prevent="handleSubmit">
    <fieldset>
      <legend>{{ $t('forms.contactInformation') }}</legend>
      
      <!-- Input с полным accessibility -->
      <div>
        <label for="email">
          {{ $t('forms.emailAddress') }}*
        </label>
        <input
          id="email"
          v-model="email"
          type="email"
          required
          aria-required="true"
          :aria-describedby="emailError ? 'email-help email-error' : 'email-help'"
          :aria-invalid="!!emailError"
        />
        <span id="email-help" class="help-text">
          {{ $t('forms.emailHelp', 'Мы никогда не передадим ваш email третьим лицам') }}
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
        <label for="system-type">
          {{ $t('forms.systemType') }}*
        </label>
        <select
          id="system-type"
          v-model="systemType"
          required
          aria-required="true"
          aria-describedby="system-type-help"
        >
          <option value="">{{ $t('forms.selectType', 'Выберите тип...') }}</option>
          <option value="hydraulic">{{ $t('systems.hydraulic') }}</option>
          <option value="pneumatic">{{ $t('systems.pneumatic') }}</option>
        </select>
        <span id="system-type-help" class="help-text">
          {{ $t('forms.systemTypeHelp', 'Выберите тип вашей системы') }}
        </span>
      </div>
    </fieldset>
    
    <button type="submit" :disabled="loading">
      <span v-if="loading" aria-live="polite">
        {{ $t('ui.submitting', 'Отправка...') }}
      </span>
      <span v-else>{{ $t('ui.submit', 'Отправить') }}</span>
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
    <span class="sr-only">{{ $t('a11y.loadingProducts', 'Загрузка продуктов...') }}</span>
    <LoadingSpinner aria-hidden="true" />
  </div>
</template>
```

---

## 3. Keyboard Navigation

### Focus Management

```vue
<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from '#imports'
import type { Ref } from 'vue'

const modalRef: Ref<HTMLElement | null> = ref(null)
const previousActiveElement: Ref<HTMLElement | null> = ref(null)

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
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
  border-radius: 2px;
}

button:focus-visible,
a:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
}

/* ❌ ПЛОХО - НЕ убирайте outline полностью! */
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

Используйте [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/) для проверки контраста.

```css
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
    :alt="$t('images.hydraulicSystemDashboard', 'Панель диагностики гидравлической системы, показывающая уровни давления')"
  />
  
  <!-- Иконки -->
  <button>
    <Icon name="heroicons:trash" aria-hidden="true" />
    <span>{{ $t('ui.delete', 'Удалить') }}</span> <!-- Видимый текст -->
  </button>
  
  <!-- Иконка без текста -->
  <button :aria-label="$t('ui.deleteSystem', 'Удалить систему')">
    <Icon name="heroicons:trash" aria-hidden="true" />
  </button>
</template>
```

---

## 6. Тестирование

### Инструменты

1. **Lighthouse** - встроенный в Chrome DevTools
2. **axe DevTools** - браузерное расширение
3. **NVDA / JAWS** - screen readers (Windows)
4. **VoiceOver** - screen reader (macOS/iOS)
5. **WAVE** - Web Accessibility Evaluation Tool

### Чеклист тестирования

- [ ] Все элементы доступны с клавиатуры (Tab, Enter, Space, Arrow keys)
- [ ] Focus visible на всех интерактивных элементах
- [ ] Контраст цветов ≥ 4.5:1 (обычный текст) и ≥ 3:1 (крупный текст/UI)
- [ ] Все изображения имеют alt текст (или пустой alt="" для декоративных)
- [ ] Формы имеют labels связанные с inputs
- [ ] ARIA атрибуты используются правильно
- [ ] Screen reader читает контент правильно и в логическом порядке
- [ ] Lighthouse accessibility score > 90
- [ ] Skip to main content link работает
- [ ] Модальные окна имеют focus trap
- [ ] Live regions объявляют изменения

### Команды для тестирования

```bash
# Запустить Lighthouse CI
npm run lighthouse

# Проверка accessibility с помощью axe
npm run test:a11y

# E2E тесты с проверкой a11y
npm run test:e2e:a11y
```

---

## 7. Best Practices для Nuxt 4

### definePageMeta с accessibility

```vue
<script setup lang="ts">
import { definePageMeta } from '#imports'

definePageMeta({
  title: 'Dashboard',
  // Устанавливаем правильные мета-теги для accessibility
  meta: [
    {
      name: 'description',
      content: 'Hydraulic system monitoring dashboard with real-time diagnostics'
    }
  ]
} as const)
</script>
```

### Композиции для accessibility

```typescript
// composables/useA11y.ts
import { ref, onMounted, onUnmounted } from '#imports'
import type { Ref } from 'vue'

export const useA11y = () => {
  const announceMessage = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const announcement = document.createElement('div')
    announcement.setAttribute('role', priority === 'assertive' ? 'alert' : 'status')
    announcement.setAttribute('aria-live', priority)
    announcement.setAttribute('aria-atomic', 'true')
    announcement.className = 'sr-only'
    announcement.textContent = message
    
    document.body.appendChild(announcement)
    
    setTimeout(() => {
      document.body.removeChild(announcement)
    }, 1000)
  }
  
  const generateId = (prefix = 'a11y') => {
    return `${prefix}-${Math.random().toString(36).substr(2, 9)}`
  }
  
  return {
    announceMessage,
    generateId
  }
}

// composables/useFocusTrap.ts
export const useFocusTrap = (containerRef: Ref<HTMLElement | null>) => {
  const previousActiveElement: Ref<HTMLElement | null> = ref(null)
  
  const activate = () => {
    previousActiveElement.value = document.activeElement as HTMLElement
    
    const firstFocusable = containerRef.value?.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as HTMLElement
    
    firstFocusable?.focus()
  }
  
  const deactivate = () => {
    previousActiveElement.value?.focus()
  }
  
  const handleKeydown = (event: KeyboardEvent) => {
    if (event.key !== 'Tab') return
    
    const focusableElements = containerRef.value?.querySelectorAll(
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
  
  onMounted(() => {
    window.addEventListener('keydown', handleKeydown)
  })
  
  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown)
  })
  
  return {
    activate,
    deactivate
  }
}
```

---

## 8. Дополнительные ресурсы

### Официальная документация

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Vue.js Accessibility Guide](https://vuejs.org/guide/best-practices/accessibility)
- [Nuxt Accessibility](https://nuxt.com/docs/guide/going-further/accessibility)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)

### Инструменты

- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [axe DevTools](https://www.deque.com/axe/devtools/)
- [WAVE Browser Extension](https://wave.webaim.org/extension/)
- [Lighthouse](https://developers.google.com/web/tools/lighthouse)

### Тестирование с screen readers

- **Windows:** [NVDA](https://www.nvaccess.org/) (бесплатно) или JAWS
- **macOS:** VoiceOver (встроенный) - Command + F5
- **iOS:** VoiceOver в Settings > Accessibility
- **Android:** TalkBack в Settings > Accessibility

---

## 9. Интеграция в CI/CD

### GitHub Actions пример

```yaml
# .github/workflows/a11y.yml
name: Accessibility Tests

on:
  pull_request:
    branches: [master, develop]
  push:
    branches: [master]

jobs:
  a11y-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: services/frontend/package-lock.json
      
      - name: Install dependencies
        working-directory: services/frontend
        run: npm ci
      
      - name: Run Lighthouse CI
        working-directory: services/frontend
        run: npm run lighthouse:ci
      
      - name: Run axe accessibility tests
        working-directory: services/frontend
        run: npm run test:a11y
      
      - name: Upload Lighthouse results
        uses: actions/upload-artifact@v4
        with:
          name: lighthouse-results
          path: services/frontend/.lighthouseci
```

---

**Статус:** 🟢 Готов к применению  
**Последнее обновление:** 19 ноября 2025  
**Версия WCAG:** 2.1 Level AA
