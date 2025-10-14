# Nuxt Frontend для гидравлической диагностической системы

Фронтенд приложение на Nuxt 3 для системы диагностики гидравлических систем.

## Структура проекта

```
nuxt_frontend/
├── components/          # Vue компоненты
│   ├── ReportsList.vue  # Список отчетов с экспортом CSV/JSON
│   ├── SystemForm.vue   # Форма создания/редактирования систем
│   └── FileUpload.vue   # Компонент загрузки файлов
├── composables/         # Переиспользуемая логика
│   ├── useNotifications.js  # Управление уведомлениями
│   ├── useReports.js    # Работа с отчетами
│   └── useSystems.js    # Работа с системами
├── pages/               # Страницы приложения (авто-роутинг)
│   ├── Login.vue        # Страница авторизации
│   └── index.vue        # Главная страница
├── tests/               # Тесты
│   └── composables/     # Unit тесты для composables
│       └── useSystems.test.js
├── app.vue              # Корневой компонент
├── package.json         # Зависимости проекта
├── vitest.config.js     # Конфигурация тестов
├── README.md            # Этот файл
└── TESTING.md           # Документация по тестированию
```

## Установка

```bash
# Установка зависимостей
npm install
```

## Разработка

```bash
# Запуск dev-сервера на http://localhost:3000
npm run dev
```

## Сборка для продакшена

```bash
# Сборка приложения
npm run build

# Предпросмотр продакшен-сборки
npm run preview
```

## Тестирование

Проект использует **Vitest** и **Vue Test Utils** для unit-тестирования.

### Запуск тестов

```bash
# Запуск всех тестов
npm test

# Запуск тестов в watch режиме
npm test -- --watch

# Запуск тестов с UI интерфейсом
npm run test:ui

# Запуск тестов с coverage отчетом
npm run test:coverage
```

### Структура тестов

Тесты располагаются в папке `tests/` и повторяют структуру основного кода:

```
tests/
└── composables/
    └── useSystems.test.js  # Тесты для useSystems.js
```

### Пример теста

```javascript
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useSystems } from '../../composables/useSystems'

// Mock fetch globally
global.fetch = vi.fn()

describe('useSystems', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    fetch.mockClear()
  })

  it('should fetch systems successfully', async () => {
    const mockSystems = [
      { id: 1, name: 'System 1', status: 'active' },
      { id: 2, name: 'System 2', status: 'inactive' }
    ]
    
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockSystems
    })

    const { systems, loading, error, fetchSystems } = useSystems()
    await fetchSystems()

    expect(systems.value).toEqual(mockSystems)
    expect(loading.value).toBe(false)
    expect(error.value).toBeNull()
  })
})
```

### Coverage отчет

Для генерации coverage отчета используйте:

```bash
npm run test:coverage
```

Отчет будет сгенерирован в папке `coverage/` в форматах:
- HTML отчет: `coverage/index.html`
- JSON отчет: `coverage/coverage-final.json`
- Текстовый отчет в консоли

Подробнее о тестировании см. [TESTING.md](./TESTING.md)

## MVP: Экспорт отчетов (CSV/JSON)

Готовая реализация экспорта отчетов доступна в компоненте `components/ReportsList.vue`.

### Возможности:

- Кнопка «Экспорт» рядом с каждым отчетом
- Выпадающее меню форматов: CSV, JSON
- Метод `downloadReport(reportId, format)` отправляет запрос `GET /systems/:systemId/reports/:reportId/export/?format=csv|json`
- Блокировка повторных кликов во время экспорта (per-report)
- Индикаторы состояния («Экспорт…») и toast-уведомления (успех/ошибка)
- Таймаут 60 сек, обработка ошибок с выводом деталей
- Инициирование скачивания через `Blob` + `URL.createObjectURL` + `a[download]`

### Требования окружения:

- В `nuxt.config` должна быть настроена переменная `runtimeConfig.public.apiUrl`, например:

```js
export default defineNuxtConfig({
  runtimeConfig: {
    public: {
      apiUrl: process.env.API_URL || 'http://localhost:8000/api/v1'
    }
  }
})
```

### Пример использования:

```vue
<template>
  <div>
    <h2>Отчеты</h2>
    <ReportsList :system-id="systemId" />
  </div>
</template>

<script setup>
import ReportsList from '~/components/ReportsList.vue'

const systemId = ref(1)
</script>
```

## Управление уведомлениями (useNotifications)

Приложение включает composable `useNotifications` для централизованного управления toast-уведомлениями.

### Возможности:

- Автоматическое удаление уведомлений через 5 секунд
- Типизированные уведомления: success, error, warning, info
- Поддержка ручного закрытия
- Реактивный список уведомлений
- Автоматическая очистка при unmount компонента

### Пример использования:

```vue
<template>
  <div>
    <!-- Container для уведомлений -->
    <div class="notifications-container">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        :class="['notification', `notification-${notification.type}`]"
      >
        {{ notification.message }}
        <button @click="removeNotification(notification.id)">✕</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useNotifications } from '~/composables/useNotifications'

const { notifications, pushNotification, removeNotification } = useNotifications()

// Примеры использования
pushNotification('success', 'Данные успешно сохранены')
pushNotification('error', 'Ошибка при загрузке данных')
</script>
```

#### Интеграция в основные компоненты:

Уведомления реализованы в следующих компонентах:

1. **ReportsList.vue** - уведомления при:
   - Успешном создании отчета
   - Ошибке создания отчета
   - Успешном экспорте (CSV/JSON)
   - Ошибке экспорта

2. **SystemForm.vue** - уведомления при:
   - Успешном создании системы
   - Успешном обновлении системы
   - Ошибках валидации
   - Ошибках сохранения

3. **FileUpload.vue** - уведомления при:
   - Успешной загрузке файла
   - Ошибке загрузки (размер, формат, сеть)
   - Прогрессе загрузки больших файлов

4. **Login.vue** - уведомления при:
   - Успешной авторизации
   - Ошибке авторизации (неверные данные)
   - Проблемах с подключением к серверу

#### API composable:

```js
const { 
  notifications,      // Ref<array> - реактивный массив активных уведомлений
  pushNotification,   // (type: string, message: string) => void
  removeNotification  // (id: number) => void
} = useNotifications()
```

#### Типы уведомлений:

- `success` - успешные операции (зеленый цвет)
- `error` - ошибки и проблемы (красный цвет)
- `warning` - предупреждения (оранжевый цвет)
- `info` - информационные сообщения (синий цвет)

## Технологии

- Nuxt 3 — Vue.js фреймворк для SSR и SSG
- Vue 3 — Progressive JavaScript Framework
- Vitest — Unit Testing Framework
- Vue Test Utils — Утилиты для тестирования Vue компонентов
- TypeScript (опционально)

## TODO

- [ ] Подключение к Django backend API
- [ ] Настройка аутентификации
- [ ] Добавление компонентов для диагностики
- [ ] Настройка Tailwind CSS / другого UI фреймворка
- [ ] Настройка state management (Pinia)
- [x] Настройка unit-тестов с Vitest
- [x] Добавление coverage отчетов

## Ссылки

- [Документация Nuxt 3](https://nuxt.com/docs)
- [Документация Vue 3](https://vuejs.org/)
- [Документация Vitest](https://vitest.dev/)
- [Документация Vue Test Utils](https://test-utils.vuejs.org/)
