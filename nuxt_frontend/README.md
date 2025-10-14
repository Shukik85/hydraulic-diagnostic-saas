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
├── app.vue              # Корневой компонент
├── package.json         # Зависимости проекта
└── README.md            # Этот файл
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

```ts
export default defineNuxtConfig({
  runtimeConfig: {
    public: {
      apiUrl: process.env.API_URL || 'http://localhost:8000/api'
    }
  }
})
```

### Пример UX-потока:

1) Пользователь нажимает «Экспорт» у нужного отчета
2) Открывается меню форматов → выбирает CSV или JSON
3) Показывается уведомление «Подготовка файла к экспорту…», кнопки блокируются от дублей
4) По завершении — «Экспорт завершен» и начинается скачивание файла `report_<id>.<format>`
5) В случае ошибок показывается уведомление с деталями; меню закрывается, блокировки снимаются

## Работа с уведомлениями в UI

### Composable: useNotifications

В проекте реализован простой и удобный механизм уведомлений через composable `useNotifications.js`.

#### Основные возможности:

- **Автоматическая очистка**: уведомления исчезают через 5 секунд
- **Типизированные сообщения**: поддержка типов `success`, `error`, `warning`, `info`
- **Ручное закрытие**: возможность закрыть уведомление вручную
- **Реактивность**: полная интеграция с Vue 3 Composition API

#### Использование в компонентах:

```vue
<script setup>
import { useNotifications } from '~/composables/useNotifications'

const { notifications, pushNotification, removeNotification } = useNotifications()

// Пример использования при успешной операции
const handleSuccess = () => {
  pushNotification('success', 'Операция выполнена успешно!')
}

// Пример использования при ошибке
const handleError = (error) => {
  pushNotification('error', `Ошибка: ${error.message}`)
}
</script>

<template>
  <!-- Отображение уведомлений -->
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
</template>
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
  notifications,      // Ref<Array> - реактивный массив активных уведомлений
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
- TypeScript (опционально)

## TODO

- [ ] Подключение к Django backend API
- [ ] Настройка аутентификации
- [ ] Добавление компонентов для диагностики
- [ ] Настройка Tailwind CSS / другого UI фреймворка
- [ ] Настройка state management (Pinia)

## Ссылки

- [Документация Nuxt 3](https://nuxt.com/docs)
- [Документация Vue 3](https://vuejs.org/)
