# Development Mode Guide

## 🚀 Quick Start

### 1. Enable Dev Mode (Bypass Authentication)

Для тестирования интерфейса без необходимости авторизации:

```bash
cd services/frontend

# Создай файл .env
cp .env.example .env

# Открой .env и измени:
NUXT_PUBLIC_DEV_SKIP_AUTH=true
```

### 2. Установка зависимостей

```bash
npm install
```

### 3. Запуск dev сервера

```bash
npm run dev
```

Приложение будет доступно на: **http://localhost:3000**

---

## 📄 Доступные страницы

### Главная страница
**URL:** http://localhost:3000

- Hero секция с описанием платформы
- 3 KPI карточки (Real-time Monitoring, AI Predictions, Systems Monitored)
- Кнопки "Get Started" и "View Dashboard"
- Информационная секция с features

### Login
**URL:** http://localhost:3000/login

- Форма входа с email и password
- Валидация полей (Zod)
- "Forgot password?" ссылка
- "Sign up" ссылка
- **Dev mode:** Можно просматривать без редиректа

### Dashboard
**URL:** http://localhost:3000/dashboard

- Заголовок и описание
- 3 KPI карточки (Active Systems, Alerts, Uptime)
- Quick Actions (Start Diagnosis, Manage Systems, View Reports)
- Recent Activity (placeholder)
- **Dev mode:** Доступен без авторизации

---

## ⚙️ Как работает Dev Mode

### Что происходит при `NUXT_PUBLIC_DEV_SKIP_AUTH=true`:

1. **Auth middleware (`middleware/auth.ts`)**
   - Пропускает проверку авторизации
   - Выводит предупреждение в консоль браузера
   - Позволяет доступ к защищённым страницам (/dashboard)

2. **Guest middleware (`middleware/guest.ts`)**
   - Пропускает проверку на авторизованного пользователя
   - Позволяет просматривать /login без редиректа

3. **Работает только в development mode**
   - Проверка `import.meta.dev` в middleware
   - **Production сборка игнорирует эту настройку**

### Консольные предупреждения

В консоли браузера ты увидишь:

```
[DEV MODE] Auth middleware bypassed - NUXT_PUBLIC_DEV_SKIP_AUTH is enabled
[DEV MODE] Guest middleware bypassed - NUXT_PUBLIC_DEV_SKIP_AUTH is enabled
```

Это нормально! Значит dev mode работает корректно.

---

## 🔒 Безопасность

### ⚠️ ВАЖНО:

- **Никогда не устанавливай `NUXT_PUBLIC_DEV_SKIP_AUTH=true` в production!**
- Эта настройка работает только в dev mode (`npm run dev`)
- Production build (`npm run build`) игнорирует dev mode настройки
- Middleware проверяет `import.meta.dev` перед bypass

### Проверка безопасности:

```bash
# Production build
npm run build
npm run preview

# Даже если NUXT_PUBLIC_DEV_SKIP_AUTH=true,
# auth middleware будет работать в production mode
```

---

## 🧪 Тестирование интерфейса

### С включённым Dev Mode:

```bash
# 1. Главная страница
open http://localhost:3000

# 2. Login (без редиректа)
open http://localhost:3000/login

# 3. Dashboard (без авторизации)
open http://localhost:3000/dashboard
```

### Тестирование Toast уведомлений:

Открой консоль браузера (F12) и выполни:

```javascript
// Success toast
const toast = useToast();
toast.success('Operation completed', 'Success');

// Error toast
toast.error('Something went wrong', 'Error');

// Warning toast
toast.warning('Please check your input', 'Warning');

// Info toast
toast.info('New feature available', 'Info');
```

---

## 🐛 Troubleshooting

### Проблема: Auth middleware не пропускает

**Симптомы:** Редирект на /login при попытке открыть /dashboard

**Решение:**

1. Проверь `.env` файл:
   ```bash
   cat .env
   # Должно быть: NUXT_PUBLIC_DEV_SKIP_AUTH=true
   ```

2. Перезапусти dev сервер:
   ```bash
   # Ctrl+C для остановки
   npm run dev
   ```

3. Проверь консоль браузера:
   - Должно быть предупреждение: `[DEV MODE] Auth middleware bypassed`
   - Если нет — значит настройка не применилась

### Проблема: Toast уведомления не отображаются

**Симптомы:** При клике на кнопки нет всплывающих уведомлений

**Решение:**

1. Проверь, что `SharedToastContainer` подключён в `app.vue`
2. Проверь консоль браузера на ошибки
3. Убедись, что `@nuxt/icon` установлен:
   ```bash
   npm install @nuxt/icon @iconify-json/heroicons
   ```

### Проблема: Иконки не отображаются

**Симптомы:** Вместо иконок пустые квадраты или ошибки

**Решение:**

```bash
# Установи пакеты иконок
npm install @nuxt/icon @iconify-json/heroicons

# Перезапусти dev сервер
npm run dev
```

### Проблема: TypeScript ошибки

**Симптомы:** Красные подчёркивания в IDE, ошибки компиляции

**Решение:**

```bash
# Сгенерируй Nuxt types
npm run dev  # Запустит и сгенерирует .nuxt/

# Или вручную
npm run postinstall

# Проверь types
npm run typecheck
```

---

## 📝 Полезные команды

```bash
# Development сервер
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint
npm run lint:fix

# Formatting
npm run format
npm run format:check

# Testing
npm run test:unit
npm run test:e2e

# Production build
npm run build
npm run preview
```

---

## 🎨 Компоненты UI

### Доступные компоненты:

- **Button** (`components/ui/Button.vue`)
  - Variants: primary, secondary, outline, ghost, destructive
  - Sizes: sm, md, lg
  - Loading state
  - Icon support

- **Input** (`components/ui/Input.vue`)
  - Types: text, email, password, number, tel, url, search
  - Error handling
  - Icon support
  - Validation

- **KpiCard** (`components/shared/KpiCard.vue`)
  - Label, value, icon
  - Trend indicator
  - Status (success, warning, error, neutral)
  - Subtext

- **ToastContainer** (`components/shared/ToastContainer.vue`)
  - Success, Error, Warning, Info variants
  - Auto-dismiss
  - Dismissible
  - ARIA live region

### Пример использования:

```vue
<template>
  <div>
    <!-- Button -->
    <Button variant="primary" size="lg" @click="handleClick">
      Click Me
    </Button>

    <!-- Input -->
    <Input
      v-model="email"
      type="email"
      label="Email"
      :error="errors.email"
      icon="heroicons:envelope"
    />

    <!-- KPI Card -->
    <KpiCard
      label="Active Users"
      value="1,234"
      :trend="12.5"
      icon="heroicons:users"
      status="success"
      subtext="Last 30 days"
    />
  </div>
</template>
```

---

## 📚 Дополнительная документация

- [README.md](./README.md) - Общая информация о проекте
- [SETUP.md](./SETUP.md) - Инструкции по установке
- [CONFIGURATION_FIXES.md](./CONFIGURATION_FIXES.md) - Исправления конфигурации

---

## 💬 Вопросы?

Если что-то не работает:

1. Проверь консоль браузера (F12) на ошибки
2. Проверь terminal где запущен `npm run dev`
3. Убедись, что все зависимости установлены: `npm install`
4. Перезапусти dev сервер

**Happy coding! 🚀**
