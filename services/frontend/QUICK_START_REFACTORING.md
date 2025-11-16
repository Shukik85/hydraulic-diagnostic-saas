# 🚀 Быстрый старт: Рефакторинг Nuxt4

**Дата:** 17 ноября 2025

---

## ✅ Что уже сделано

### Базовые компоненты созданы:

```
services/frontend/
├── components/ui/
│   ├── UZeroState.vue       ✅ Пустые состояния
│   ├── UStatusDot.vue       ✅ Индикаторы статуса
│   ├── UHelperText.vue      ✅ Helper тексты
│   ├── UFormGroup.vue       ✅ Обертка форм
│   └── UGauge.vue           ✅ Gauge индикатор
└── styles/
    └── components.css      ✅ Утилитарные классы
```

---

## 📝 Следующие шаги

### 1. Импортировать components.css

В `app.vue` или `nuxt.config.ts` добавьте:

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  css: [
    '~/styles/metallic.css',
    '~/styles/premium-tokens.css',
    '~/styles/components.css', // ← Новый файл
  ],
})
```

### 2. Добавить Zero State в Diagnostics

**Файл:** `pages/diagnostics/index.vue`

```vue
<template>
  <div>
    <!-- ... existing code ... -->
    
    <!-- Добавьте перед списком -->
    <UZeroState
      v-if="!loading && diagnostics.length === 0"
      icon-name="heroicons:document-magnifying-glass"
      title="Нет активных диагностик"
      description="Запустите первую диагностику для анализа гидравлической системы"
      action-icon="heroicons:play"
      action-text="Запустить диагностику"
      @action="openRunDiagnosticModal"
    />

    <!-- Существующий список -->
    <div v-else>
      <!-- ... existing list ... -->
    </div>
  </div>
</template>
```

### 3. Добавить Zero State в Systems

**Файл:** `pages/systems/index.vue`

```vue
<template>
  <div>
    <UZeroState
      v-if="!loading && systems.length === 0"
      icon-name="heroicons:cube"
      title="Системы не добавлены"
      description="Добавьте первую гидравлическую систему для мониторинга"
      action-icon="heroicons:plus"
      action-text="Добавить систему"
      @action="openCreateSystemModal"
    />

    <!-- Список систем -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <UCard 
        v-for="system in systems" 
        :key="system.id"
        class="card-interactive"
      >
        <UCardHeader>
          <div class="flex items-center justify-between">
            <UCardTitle>{{ system.name }}</UCardTitle>
            
            <!-- Добавьте статус индикатор -->
            <UStatusDot 
              :status="system.is_active ? 'success' : 'offline'"
              :label="system.is_active ? 'Онлайн' : 'Оффлайн'"
            />
          </div>
        </UCardHeader>
        <!-- ... rest of card ... -->
      </UCard>
    </div>
  </div>
</template>
```

### 4. Обновить формы с UFormGroup

**Пример:** `components/ui/URunDiagnosticModal.vue`

```vue
<template>
  <UModal v-model="isOpen">
    <UDialogHeader>
      <UDialogTitle>Запустить диагностику</UDialogTitle>
    </UDialogHeader>

    <form @submit.prevent="handleSubmit">
      <!-- Вместо простого input -->
      <UFormGroup
        label="Система"
        helper="Выберите систему для анализа"
        :error="errors.systemId"
        required
      >
        <USelect v-model="formData.systemId">
          <option value="" disabled>Выберите систему</option>
          <option v-for="system in systems" :key="system.id" :value="system.id">
            {{ system.name }}
          </option>
        </USelect>
      </UFormGroup>

      <UFormGroup
        label="Приоритет"
        helper="Высокий приоритет обрабатывается быстрее"
        class="mt-4"
      >
        <USelect v-model="formData.priority">
          <option value="low">Низкий</option>
          <option value="medium">Средний</option>
          <option value="high">Высокий</option>
        </USelect>
      </UFormGroup>

      <UDialogFooter class="mt-6">
        <UButton type="button" variant="ghost" @click="isOpen = false">
          Отмена
        </UButton>
        <UButton type="submit" :disabled="loading">
          Запустить
        </UButton>
      </UDialogFooter>
    </form>
  </UModal>
</template>
```

### 5. Использование CSS классов

```vue
<!-- Кнопки -->
<button class="btn-primary">
  Основная кнопка
</button>

<button class="btn-primary-lg">
  Большая кнопка
</button>

<button class="btn-secondary">
  Вторичная
</button>

<button class="btn-icon">
  <Icon name="heroicons:cog-6-tooth" />
</button>

<!-- Карточки -->
<div class="card-glass p-6">
  Контент карточки
</div>

<div class="card-interactive p-6" @click="handleClick">
  Интерактивная карточка
</div>

<!-- Инпуты -->
<input 
  class="input-text" 
  placeholder="Введите текст..."
/>

<!-- Бейджи -->
<span class="badge-success">
  <Icon name="heroicons:check" class="w-3 h-3" />
  Активно
</span>

<span class="badge-warning">
  <Icon name="heroicons:exclamation-triangle" class="w-3 h-3" />
  Предупреждение
</span>

<!-- Алерты -->
<div class="alert-success">
  <Icon name="heroicons:check-circle" class="w-5 h-5" />
  <div>
    <strong>Успех!</strong>
    <p>Операция выполнена успешно</p>
  </div>
</div>
```

---

## 🛠️ Полезные команды

### Поиск emoji для замены:

```bash
# Найти все emoji в проекте
grep -r "💡\|✅\|⚠️\|❌\|🔴\|🟢" pages/ components/ --include="*.vue"

# Найти кнопки без указанного размера
grep -r "<UButton" pages/ components/ --include="*.vue" | grep -v 'size="'

# Найти формы без helper текста
grep -r "<UInput\|<USelect\|<UTextarea" components/ --include="*.vue" -A 5 | grep -v "helper"
```

### Запуск проекта:

```bash
cd services/frontend
npm install
npm run dev
```

### Проверка линтера:

```bash
npm run lint
npm run lint:fix
```

### TypeScript check:

```bash
npx nuxi typecheck
```

---

## 📚 Справочник Heroicons

### Часто используемые иконки:

```vue
<!-- Действия -->
<Icon name="heroicons:play" />              <!-- Запуск -->
<Icon name="heroicons:pause" />             <!-- Пауза -->
<Icon name="heroicons:stop" />              <!-- Остановка -->
<Icon name="heroicons:plus" />              <!-- Добавить -->
<Icon name="heroicons:x-mark" />            <!-- Закрыть -->
<Icon name="heroicons:trash" />             <!-- Удалить -->
<Icon name="heroicons:pencil" />            <!-- Редактировать -->

<!-- Статусы -->
<Icon name="heroicons:check-circle" />      <!-- Успех -->
<Icon name="heroicons:x-circle" />          <!-- Ошибка -->
<Icon name="heroicons:exclamation-triangle" /> <!-- Предупреждение -->
<Icon name="heroicons:information-circle" /> <!-- Информация -->

<!-- Функционал -->
<Icon name="heroicons:magnifying-glass" />  <!-- Поиск -->
<Icon name="heroicons:cog-6-tooth" />       <!-- Настройки -->
<Icon name="heroicons:chart-bar" />         <!-- Графики -->
<Icon name="heroicons:document-text" />     <!-- Документ -->
<Icon name="heroicons:folder" />            <!-- Папка -->
<Icon name="heroicons:arrow-down-tray" />   <!-- Скачать -->
<Icon name="heroicons:arrow-up-tray" />     <!-- Загрузить -->

<!-- Специальные -->
<Icon name="heroicons:light-bulb" />        <!-- Совет -->
<Icon name="heroicons:rocket-launch" />     <!-- Запуск -->
<Icon name="heroicons:cube" />              <!-- Система -->
<Icon name="heroicons:chat-bubble-left-right" /> <!-- Чат -->
```

Полный список: https://heroicons.com/

---

## ❓ Частые вопросы

### Как изменить цвет UZeroState?

Используйте prop `variant`:

```vue
<UZeroState
  variant="success"    <!-- success | warning | error | info -->
  icon-name="heroicons:check-circle"
  title="Все готово!"
  description="Операция завершена"
  :show-action="false"
/>
```

### Как отключить анимацию UStatusDot?

```vue
<UStatusDot 
  status="success" 
  :animated="false" 
/>
```

### Как использовать UGauge с нестандартными цветами?

```vue
<UGauge
  :value="75"
  :max="100"
  unit="%"
  label="Производительность"
  color="#10b981"  <!-- Кастомный цвет -->
  bg-color="#1e293b"
/>
```

---

## 👥 Помощь

Если возникли вопросы:

1. Прочитайте [REFACTORING_PLAN.md](./REFACTORING_PLAN.md)
2. Изучите [FRIENDLY_UI_UX_GUIDE.md](./FRIENDLY_UI_UX_GUIDE.md)
3. Посмотрите [DESIGN_AUDIT_PLAN.md](./DESIGN_AUDIT_PLAN.md)

---

**Удачи в рефакторинге! 🚀**
