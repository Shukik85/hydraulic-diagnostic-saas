# Hydraulic Diagnostic SaaS - Frontend

Профессиональная frontend платформа для системы диагностики гидравлических систем, построенная на Nuxt 4 с современным стеком технологий 2025 года.

## 🚀 Быстрый старт

### Требования
- **Node.js:** 18.x или 20.x LTS (рекомендуется 20.x)
- **NPM:** ≥ 8.0 (или PNPM/Yarn при необходимости)
- **Python:** 3.11+ (для работы с backend API)

### Установка и запуск

1. **Установка зависимостей:**
   ```bash
   cd nuxt_frontend
   npm ci
   ```
   > Используйте `npm ci` для воспроизводимых установок из package-lock.json

2. **Настройка окружения:**
   
   Создайте `.env.local` (опционально):
   ```env
   NUXT_PUBLIC_API_BASE=http://localhost:8000/api
   NUXT_PUBLIC_SITE_URL=http://localhost:3000
   ```
   
   > Переменные уже настроены в nuxt.config.ts с дефолтными значениями

3. **Запуск в режиме разработки:**
   ```bash
   npm run dev
   ```
   
   Приложение будет доступно на: http://localhost:3000

## 📦 Доступные команды

### Разработка
- `npm run dev` - запуск dev-сервера с hot-reload
- `npm run build` - production сборка
- `npm run preview` - preview production сборки
- `npm run generate` - статическая генерация (SSG)

### Качество кода
- `npm run lint` - проверка ESLint
- `npm run lint:check` - только проверка без исправлений  
- `npm run format` - форматирование кода Prettier
- `npm run format:check` - проверка форматирования
- `npm run typecheck` - проверка типов TypeScript

### Анализ и утилиты
- `npm run analyze` - анализ размера бандла
- `npm run clean` - очистка кэшей и артефактов

## 🏗️ Архитектура

### Структура проекта
```
nuxt_frontend/
├── pages/           # Страницы приложения (file-based роутинг)
├── layouts/         # Макеты страниц
├── components/      # Vue компоненты
├── composables/     # Композабли (логика переиспользования)
├── stores/          # Pinia состояние (user, auth и тд)
├── plugins/         # Nuxt плагины
├── middleware/      # Middleware (авторизация и тд)
├── types/           # TypeScript типы
├── assets/          # Ресурсы для сборки
├── styles/          # Глобальные стили
├── public/          # Статичные файлы
└── trash/           # Перемещённые файлы (безопасно удаляемые)
```

### Ключевые технологии
- **Nuxt 4** - Meta framework для Vue.js
- **Vue 3** - Reactive UI framework с Composition API
- **TypeScript** - Типизированный JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Pinia** - State management для Vue
- **Nuxt modules:** ESLint, Image, Icon, Fonts, SEO

## 🔗 Интеграция с Backend

Фронтенд интегрирован с Django REST API:

- **API клиент:** `composables/useApi.ts` - HTTP клиент с JWT авторизацией
- **Типы:** `types/api.ts` - TypeScript типы, соответствующие Django моделям
- **Авторизация:** `stores/auth.ts` - Pinia store для управления сессиями
- **Middleware:** `middleware/auth.ts` - Защита маршрутов

### Основные эндпоинты
- `/auth/*` - Авторизация (login, register, logout)
- `/users/*` - Управление пользователями
- `/systems/*` - Гидравлические системы (CRUD)
- `/reports/*` - Диагностические отчёты
- `/sensors/*` - Данные сенсоров
- `/rag/*` - AI-помощник и анализ

## 🎨 UI/UX Features

### Дизайн система
- **Responsive design** - Mobile-first подход
- **Dark/Light режимы** - Автоматическое переключение
- **Компонентная библиотека** - Переиспользуемые UI компоненты
- **Анимации** - Плавные микровзаимодействия
- **Accessibility** - WCAG 2.1 AA совместимость

### Ключевые страницы
- `/` - Landing page для инвесторов и клиентов
- `/dashboard` - Основная рабочая панель
- `/investors` - Специальная страница для инвесторов
- `/equipment/*` - Управление гидравлическими системами
- `/diagnostics` - Запуск и мониторинг диагностик
- `/reports/*` - Просмотр отчётов и аналитики
- `/chat` - AI-помощник для анализа данных
- `/auth/*` - Авторизация и регистрация

## 🔧 Разработка

### Добавление новой страницы
1. Создайте файл в `pages/` - автоматически создастся роут
2. Добавьте TypeScript типы в `types/`
3. При необходимости создайте composable в `composables/`
4. Обновите навигацию в соответствующих layouts

### Работа с API
```typescript
// В компоненте
const api = useApi()
const { data, pending, error } = await useLazyAsyncData(
  'my-data',
  () => api.getSystems()
)
```

### Добавление нового store
```typescript
// stores/myStore.ts
export const useMyStore = defineStore('my-store', () => {
  // Composition API style
  const state = ref(initialValue)
  const actions = () => { /* ... */ }
  return { state, actions }
})
```

## 🚀 Продакшн

### Сборка
```bash
npm run build
```

### Переменные окружения
Установите в продакшене:
```env
NUXT_PUBLIC_API_BASE=https://your-api.domain.com/api
NUXT_PUBLIC_SITE_URL=https://your-frontend.domain.com
```

### SEO оптимизация
- ✅ Server-Side Rendering (SSR)
- ✅ OpenGraph и Twitter Cards
- ✅ Структурированные данные
- ✅ Optimized изображения через @nuxt/image
- ✅ Сжатие и кэширование статики

## 📊 Performance

- **Bundle analysis:** `npm run analyze`
- **Lighthouse:** Настроено для 90+ баллов
- **Code splitting:** Автоматический по роутам
- **Tree shaking:** Удаление неиспользуемого кода
- **Image optimization:** Webp/Avif, lazy loading

## 🔍 Troubleshooting

### Частые проблемы

1. **Порт занят:**
   ```bash
   NUXT_PORT=3001 npm run dev
   ```

2. **Проблемы с кэшем:**
   ```bash
   npm run clean
   rm -rf .nuxt node_modules/.cache
   npm ci
   ```

3. **Проблемы с типами:**
   ```bash
   npm run typecheck
   # или
   npx nuxi prepare
   ```

4. **Backend недоступен:**
   Проверьте NUXT_PUBLIC_API_BASE и что Django сервер запущен

### Логирование
- Development: Включено полное логирование
- Production: Только ошибки и warning

## 📝 Очистка проекта

В директории `trash/` находятся файлы, которые были перемещены во время очистки проекта:
- Build артефакты (.nuxt/)
- Внешние проекты (my-nuxt-app/)
- Черновики и документация
- Неиспользуемые конфиги

Эти файлы можно безопасно удалить или восстановить при необходимости.

## 🎯 Готовность к демонстрации

Проект готов для:
- ✅ **Демонстрации инвесторам** - профессиональный landing и investor dashboard
- ✅ **Production deployment** - оптимизированная сборка
- ✅ **Mobile presentation** - адаптивный дизайн
- ✅ **Live данные** - интеграция с реальным API
- ✅ **Enterprise security** - JWT авторизация, защищённые роуты

---

**Версия:** 1.0.0 (Октябрь 2025)  
**Стек:** Nuxt 4, Vue 3, TypeScript, Tailwind CSS, Pinia