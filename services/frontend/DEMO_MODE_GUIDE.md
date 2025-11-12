# 🎭 Demo Mode Guide

## Overview

Demo Mode позволяет показывать полнофункциональную демонстрацию приложения без необходимости регистрации и backend.

---

## 🚀 Quick Start

### Включить Demo Mode:

```bash
# 1. Создай .env файл
cp .env.example .env

# 2. Включи demo mode
echo "NUXT_PUBLIC_DEMO_MODE=true" >> .env

# 3. Перезапусти dev server
npm run dev
```

### Выключить Demo Mode:

```bash
# В .env
NUXT_PUBLIC_DEMO_MODE=false
```

---

## 📋 Features

### ✅ Что работает в Demo Mode:

1. **Auto-Login**
   - Автоматическая авторизация как demo user
   - Нет необходимости в credentials
   - Работает на всех страницах

2. **Mock Data**
   - 4 единицы оборудования
   - 5 alerts различной критичности
   - 4 диагностических сессии
   - 2 системы

3. **Read-Only Mode**
   - `authStore.canEdit` = false
   - Disabled кнопки редактирования
   - Fake операции (toast уведомления)

4. **Demo Banner**
   - Фиолетовый баннер сверху
   - "Exit Demo" кнопка
   - "Get Started" CTA

---

## 🎯 Use Cases

### 1. Презентации инвесторам
```bash
NUXT_PUBLIC_DEMO_MODE=true
npm run build
npm run preview
```

### 2. User testing
- Дай тестировщикам доступ без регистрации
- Собирай feedback на mock данных
- Быстрые итерации

### 3. Marketing demos
- Embed на лендинг
- "Try Demo" кнопка
- Конверсия в signup

---

## 🔧 Customization

### Изменить demo user:

```bash
# .env
NUXT_PUBLIC_DEMO_USER_NAME=John Demo
NUXT_PUBLIC_DEMO_USER_EMAIL=john@demo.com
```

### Добавить mock данные:

```typescript
// composables/useDemoData.ts
const demoEquipment = [
  {
    id: 'demo-005',
    name: 'Your Equipment',
    // ...
  },
]
```

### Изменить поведение кнопок:

```vue
<UButton
  :disabled="!authStore.canEdit"
  @click="handleEdit"
>
  Edit
</UButton>
```

---

## 🎨 Demo Banner

### Customization:

```vue
<!-- components/ui/DemoBanner.vue -->
<div class="bg-gradient-to-r from-purple-600 to-blue-600">
  <!-- Change colors, text, buttons -->
</div>
```

### Hide banner:

```vue
<!-- layouts/default.vue -->
<DemoBanner v-if="showBanner" />
```

---

## 📊 Mock Data

### Equipment (4 items):
- Excavator CAT 320D (health: 87%)
- Hydraulic Press HPM-500 (health: 92%)
- Mobile Crane LTM 1300 (health: 75%)
- Loader Volvo L350F (health: 95%)

### Alerts (5 items):
- 1 Critical
- 1 Error
- 1 Warning
- 2 Info

### Diagnostics (4 sessions):
- 3 Completed
- 1 In Progress

---

## 🔒 Security

### Demo Mode отключается в production:

```typescript
// middleware/auth.ts
if (config.public.demoMode) {
  // Only works if explicitly enabled
}
```

### Environment variables:
```bash
# Production .env
NUXT_PUBLIC_DEMO_MODE=false  # Always false!
```

---

## 🚀 Deployment

### Staging с Demo:
```bash
# .env.staging
NUXT_PUBLIC_DEMO_MODE=true
```

### Production без Demo:
```bash
# .env.production
NUXT_PUBLIC_DEMO_MODE=false
```

---

## 💡 Tips

1. **Всегда используй demo mode для презентаций**
   - Не зависит от backend
   - Быстрая загрузка
   - Предсказуемые данные

2. **Добавляй реалистичные данные**
   - Реальные названия оборудования
   - Правдоподобные метрики
   - Разнообразие статусов

3. **Тестируй toggle между modes**
   - Dev → Demo → Production
   - Проверяй что всё работает

---

## 🎯 Next Steps

1. **Add more mock data** - расширь useDemoData
2. **Customize banner** - брендируй DemoBanner
3. **Add analytics** - трекинг demo sessions
4. **Create demo video** - записывай walkthrough

---

## 📞 Support

Если что-то не работает:
1. Проверь `.env` файл
2. Перезапусти dev server
3. Проверь console.log в browser
4. Проверь `authStore.isDemoMode`

---

**Ready to demo!** 🎉
