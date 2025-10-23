# HydraulicsTell — Nuxt migration skeleton

Этот репозиторий содержит стартовый скелет для миграции вашего React (Vite) проекта в Nuxt (Vue 3 / Nuxt 4).

Коротко:
- Конфигурация: `nuxt.config.ts`
- Точки входа: `app.vue`, `layouts/default.vue`, `pages/index.vue`
- Стили: `assets/css/globals.css` (скопированы из `styles/globals.css`)

Как запустить:
1. Установите зависимости:

```powershell
npm install
```

2. Запустить dev-сервер:

```powershell
npm run dev
```

Дальше:
- Перенести компоненты из `components/` (React -> Vue)
- Настроить Tailwind (если нужен) и ESLint/Prettier
- Привести маршруты из `pages/` React к файловой маршрутизации Nuxt
