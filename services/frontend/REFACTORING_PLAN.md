# 🎨 План рефакторинга Nuxt4 приложения

**Дата:** 17 ноября 2025  
**Ветка:** `fix/frontend-audit-nuxt4`  
**Статус:** В процессе

---

## ✅ Батч 1: Базовые UI компоненты (ВЫПОЛНЕНО)

### Созданные компоненты:

1. **UZeroState.vue** - универсальный компонент для пустых состояний
   - Props: iconName, title, description, actionText, actionIcon, variant
   - Использование: Diagnostics, Systems, Reports, Chat

2. **UStatusDot.vue** - индикатор статуса с анимацией
   - Props: status (success/warning/error/info/offline), label, animated
   - Использование: Systems list, Sensors

3. **UHelperText.vue** - helper текст для форм
   - Props: text, variant, icon, showIcon
   - Использование: все формы

4. **UFormGroup.vue** - обертка для полей форм
   - Props: label, helper, error, required, inputId
   - Использование: все формы

5. **UGauge.vue** - круговой gauge индикатор
   - Props: value, max, min, unit, label, color
   - Использование: Sensors, Dashboard

6. **components.css** - утилитарные классы
   - Классы кнопок (btn-primary, btn-secondary, btn-ghost, btn-icon)
   - Классы карточек (card-glass, card-hover, card-interactive)
   - Классы форм (input-text, select-custom, textarea-custom)
   - Классы алертов (alert-success, alert-warning, alert-error, alert-info)
   - Классы бейджей (badge-success, badge-warning, badge-error, badge-info)
   - Progress bars, skeletons, helpers

---

## 📋 Батч 2: Рефакторинг страниц с Zero States

### Приоритет: 🔴 ВЫСОКИЙ

### 2.1. Diagnostics Page (`pages/diagnostics/index.vue`)

**Задача:** Добавить Zero State для пустого списка диагностик

```vue
<!-- Добавить в секцию, где отображается список -->
<UZeroState
  v-if="diagnostics.length === 0"
  icon-name="heroicons:document-magnifying-glass"
  :title="$t('diagnostics.empty.title')"
  :description="$t('diagnostics.empty.description')"
  action-icon="heroicons:play"
  :action-text="$t('diagnostics.empty.action')"
  @action="openRunDiagnosticModal"
/>
```

**i18n ключи добавить:**
```json
{
  "diagnostics": {
    "empty": {
      "title": "Нет активных диагностик",
      "description": "Запустите первую диагностику для анализа гидравлической системы",
      "action": "Запустить диагностику"
    }
  }
}
```

### 2.2. Systems Page (`pages/systems/index.vue`)

**Задача:** Добавить Zero State + Status Dots

```vue
<!-- Zero State -->
<UZeroState
  v-if="systems.length === 0"
  icon-name="heroicons:cube"
  :title="$t('systems.empty.title')"
  :description="$t('systems.empty.description')"
  action-icon="heroicons:plus"
  :action-text="$t('systems.empty.action')"
  @action="openCreateSystemModal"
/>

<!-- В карточках систем добавить статус -->
<div class="flex items-center justify-between mb-2">
  <h3 class="text-lg font-bold">{{ system.name }}</h3>
  <UStatusDot 
    :status="system.is_active ? 'success' : 'offline'"
    :label="system.is_active ? 'Онлайн' : 'Оффлайн'"
  />
</div>
```

### 2.3. Reports Page (`pages/reports/index.vue`)

```vue
<UZeroState
  v-if="reports.length === 0"
  icon-name="heroicons:document-text"
  :title="$t('reports.empty.title')"
  :description="$t('reports.empty.description')"
  action-icon="heroicons:document-plus"
  :action-text="$t('reports.empty.action')"
  @action="openGenerateReportModal"
/>
```

### 2.4. Chat Page (`pages/chat.vue`)

```vue
<!-- Приветственное сообщение вместо пустого чата -->
<div v-if="messages.length === 0" class="flex flex-col items-center justify-center h-full py-20">
  <div class="max-w-2xl text-center">
    <Icon name="heroicons:chat-bubble-left-right" class="w-16 h-16 text-primary-400 mx-auto mb-6" />
    <h2 class="text-2xl font-bold text-white mb-4">
      {{ $t('chat.welcome.title') }}
    </h2>
    <p class="text-steel-shine mb-8">
      {{ $t('chat.welcome.description') }}
    </p>

    <!-- Примеры вопросов -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      <button
        v-for="example in exampleQuestions"
        :key="example.id"
        class="p-4 rounded-lg bg-steel-800/50 border border-steel-700/50 
               hover:border-primary-500/50 hover:bg-steel-800/80 transition-all
               text-left text-sm text-steel-100"
        @click="askQuestion(example.text)"
      >
        <Icon :name="example.icon" class="w-5 h-5 text-primary-400 mb-2" />
        {{ example.text }}
      </button>
    </div>
  </div>
</div>
```

---

## 📋 Батч 3: Улучшение форм с Helper Text

### Приоритет: 🔴 ВЫСОКИЙ

### 3.1. Dashboard Modals

**URunDiagnosticModal.vue:**

```vue
<UFormGroup
  :label="$t('diagnostics.form.system')"
  helper="Выберите систему для анализа"
  required
>
  <USelect v-model="formData.systemId">
    <option v-for="system in systems" :key="system.id" :value="system.id">
      {{ system.name }}
    </option>
  </USelect>
</UFormGroup>

<UFormGroup
  :label="$t('diagnostics.form.priority')"
  helper="Высокий приоритет обрабатывается быстрее"
>
  <USelect v-model="formData.priority">
    <option value="low">Низкий</option>
    <option value="medium">Средний</option>
    <option value="high">Высокий</option>
  </USelect>
</UFormGroup>
```

**UCreateSystemModal.vue:**

```vue
<UFormGroup
  :label="$t('systems.form.name')"
  helper="Используйте понятное имя для идентификации"
  :error="errors.name"
  required
>
  <UInput 
    v-model="formData.name" 
    placeholder="Например: Гидравлическая система №1"
  />
</UFormGroup>

<UFormGroup
  :label="$t('systems.form.description')"
  helper="Опишите назначение и особенности системы"
>
  <UTextarea 
    v-model="formData.description" 
    placeholder="Краткое описание..."
  />
</UFormGroup>
```

**UReportGenerateModal.vue:**

```vue
<UFormGroup
  :label="$t('reports.form.format')"
  helper="PDF - для печати, Excel - для анализа данных"
  required
>
  <URadioGroup v-model="formData.format">
    <URadioGroupItem value="pdf" label="PDF" />
    <URadioGroupItem value="excel" label="Excel" />
  </URadioGroup>
</UFormGroup>
```

### 3.2. Settings Forms

**pages/settings/profile.vue:**

```vue
<UFormGroup
  label="Email"
  helper="Используется для уведомлений и восстановления пароля"
  :error="errors.email"
  required
>
  <UInput 
    v-model="profile.email" 
    type="email"
    placeholder="user@example.com"
  />
</UFormGroup>
```

---

## 📋 Батч 4: Замена Emoji на SVG иконки

### Приоритет: 🔴 ВЫСОКИЙ

### Массовая замена по всем файлам:

```bash
# Поиск всех emoji
grep -r "💡\|✅\|⚠️\|❌\|🔴\|🟢\|⚙️\|📊\|📈\|🚀" pages/ components/ --include="*.vue"
```

### Маппинг замены:

```javascript
const emojiToIconMap = {
  '💡': 'heroicons:light-bulb',
  '✅': 'heroicons:check-circle',
  '⚠️': 'heroicons:exclamation-triangle',
  '❌': 'heroicons:x-circle',
  '🔴': 'heroicons:x-circle',
  '🟢': 'heroicons:check-circle',
  '⚙️': 'heroicons:cog-6-tooth',
  '📊': 'heroicons:chart-bar',
  '📈': 'heroicons:chart-bar-square',
  '🚀': 'heroicons:rocket-launch',
  '📝': 'heroicons:document-text',
  '🔧': 'heroicons:wrench-screwdriver',
  '💾': 'heroicons:archive-box',
  '📁': 'heroicons:folder',
  '🔍': 'heroicons:magnifying-glass',
  '🎯': 'heroicons:cursor-arrow-rays',
  '📤': 'heroicons:arrow-up-tray',
  '📥': 'heroicons:arrow-down-tray',
}
```

### Пример замены в компоненте:

```vue
<!-- ДО -->
<span>💡 Совет</span>

<!-- ПОСЛЕ -->
<div class="flex items-center gap-2">
  <Icon name="heroicons:light-bulb" class="w-5 h-5 text-primary-400" />
  <span>Совет</span>
</div>
```

---

## 📋 Батч 5: Увеличение кнопок до 48px+

### Приоритет: 🔴 ВЫСОКИЙ

### Глобальная замена в UButton.vue:

```vue
<!-- Обновить размеры в UButton.vue -->
<script setup lang="ts">
const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-lg font-medium transition-all',
  {
    variants: {
      size: {
        default: 'h-12 px-6 py-3',      // 48px
        sm: 'h-10 px-4 py-2',           // 40px (минимум для mobile)
        lg: 'h-14 px-8 py-4 text-lg',  // 56px
        icon: 'h-12 w-12',              // 48x48
      },
    },
    defaultVariants: {
      size: 'default',
    },
  }
)
</script>
```

### Обновить все кнопки в проекте:

```bash
# Найти все использования UButton без size
grep -r "<UButton" pages/ components/ --include="*.vue" | grep -v "size="
```

---

## 📋 Батч 6: Интеграция Gauge в Sensors

### Приоритет: 🟠 СРЕДНИЙ

**pages/sensors.vue:**

```vue
<template>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    <UCard 
      v-for="sensor in sensors" 
      :key="sensor.id"
      class="card-glass"
    >
      <UCardHeader>
        <div class="flex items-center justify-between">
          <UCardTitle>{{ sensor.name }}</UCardTitle>
          <UStatusDot 
            :status="getSensorStatus(sensor.value, sensor.threshold)"
            :label="sensor.unit"
          />
        </div>
      </UCardHeader>

      <UCardContent>
        <!-- Gauge Visualization -->
        <UGauge
          :value="sensor.value"
          :max="sensor.max_value"
          :min="sensor.min_value"
          :unit="sensor.unit"
          :label="sensor.description"
          show-status
          :status-thresholds="{
            success: sensor.threshold_high,
            warning: sensor.threshold_low
          }"
        />

        <!-- History Chart -->
        <div class="mt-6">
          <h4 class="text-sm font-medium text-steel-shine mb-3">
            История показаний (24ч)
          </h4>
          <chart-line 
            :data="sensor.history" 
            :height="100"
            class="w-full"
          />
        </div>
      </UCardContent>

      <UCardFooter>
        <div class="flex justify-between items-center text-xs text-steel-400">
          <span>Последнее обновление</span>
          <span>{{ formatDate(sensor.last_updated) }}</span>
        </div>
      </UCardFooter>
    </UCard>
  </div>
</template>
```

---

## 📋 Батч 7: Улучшение Dashboard KPI Cards

### Приоритет: 🟠 СРЕДНИЙ

**components/ui/KpiCard.vue - обновить:**

```vue
<template>
  <div class="card-glass p-6 card-hover">
    <!-- Header -->
    <div class="flex items-start justify-between mb-4">
      <div class="flex-1">
        <p class="text-sm text-steel-shine font-medium mb-1">
          {{ title }}
        </p>
        <div class="flex items-baseline gap-2">
          <span class="text-4xl font-bold text-white">
            {{ value }}
          </span>
          <span class="text-sm text-steel-400">
            {{ unit }}
          </span>
        </div>
      </div>

      <!-- Icon -->
      <div 
        class="w-12 h-12 rounded-lg flex items-center justify-center"
        :class="iconBgClass"
      >
        <Icon 
          :name="icon" 
          class="w-6 h-6"
          :class="iconColorClass"
        />
      </div>
    </div>

    <!-- Trend -->
    <div 
      v-if="trend"
      class="flex items-center gap-1.5"
      :class="trendColorClass"
    >
      <Icon 
        :name="trendIcon" 
        class="w-4 h-4"
      />
      <span class="text-sm font-medium">
        {{ trend.value }}{{ trend.unit || '%' }}
      </span>
      <span class="text-xs text-steel-400">
        {{ trend.label || 'от вчера' }}
      </span>
    </div>

    <!-- Helper Text -->
    <UHelperText 
      v-if="helper"
      :text="helper"
      class="mt-3"
    />
  </div>
</template>
```

---

## 📋 Батч 8: Accessibility Improvements

### Приоритет: 🟡 НИЗКИЙ

### 8.1. Добавить aria-labels

```vue
<!-- Кнопки без текста -->
<UButton 
  variant="ghost" 
  size="icon"
  aria-label="Закрыть"
>
  <Icon name="heroicons:x-mark" />
</UButton>

<!-- Формы -->
<UInput 
  v-model="search"
  aria-label="Поиск систем"
  placeholder="Поиск..."
/>
```

### 8.2. Keyboard Navigation

```vue
<!-- Добавить @keydown.enter -->
<div 
  class="card-interactive"
  role="button"
  tabindex="0"
  @click="openSystem(system)"
  @keydown.enter="openSystem(system)"
>
  <!-- content -->
</div>
```

### 8.3. Focus States

```css
/* В components.css добавлено focus:ring-2 для всех интерактивных элементов */
```

---

## 🗓️ Timeline Implementation

### Неделя 1 (18-24 ноября):
- ✅ День 1: Базовые UI компоненты (Батч 1) - ГОТОВО
- День 2: Zero States (Батч 2) - pages/diagnostics, pages/systems
- День 3: Zero States (Батч 2) - pages/reports, pages/chat
- День 4: Helper Text (Батч 3) - Dashboard modals
- День 5: Emoji → SVG (Батч 4) - массовая замена

### Неделя 2 (25 ноября - 1 декабря):
- День 1: Helper Text (Батч 3) - Settings forms
- День 2: Увеличение кнопок (Батч 5)
- День 3-4: Gauge integration (Батч 6) - Sensors page
- День 5: KPI Cards (Батч 7) - Dashboard improvements

### Неделя 3 (2-8 декабря):
- День 1-2: Accessibility (Батч 8)
- День 3: Testing & Bug fixes
- День 4: Documentation update
- День 5: Final QA & review

---

## 📊 Метрики успеха

| Метрика | До | Цель | Текущее |
|---------|-----|------|---------|
| UI/UX Score | 6/10 | 9/10 | 7/10 |
| Zero States | 0/4 | 4/4 | 0/4 |
| Helper Text | 0/15 | 15/15 | 0/15 |
| Emoji → SVG | 0% | 100% | 0% |
| Button Size | 50% | 100% | 50% |
| Accessibility | 5/10 | 9/10 | 5/10 |

---

## 🔗 Связанные файлы

- [DESIGN_AUDIT_PLAN.md](./DESIGN_AUDIT_PLAN.md)
- [FRIENDLY_UI_UX_GUIDE.md](./FRIENDLY_UI_UX_GUIDE.md)
- [ROADMAP.md](./ROADMAP.md)
- [UI_UX_PAGES_SUMMARY.md](./UI_UX_PAGES_SUMMARY.md)
- [TAILWIND_CSS_CLASSES.md](./TAILWIND_CSS_CLASSES.md)

---

## ✅ Чеклист следующих действий

- [ ] Проверить работу созданных компонентов
- [ ] Добавить Zero State в Diagnostics
- [ ] Добавить Zero State в Systems  
- [ ] Добавить Zero State в Reports
- [ ] Добавить Zero State в Chat
- [ ] Обновить все формы с UFormGroup
- [ ] Массовая замена emoji на иконки
- [ ] Увеличить размер кнопок
- [ ] Интегрировать UGauge в Sensors
- [ ] Обновить KPI Cards в Dashboard
- [ ] Добавить aria-labels
- [ ] Протестировать keyboard navigation
