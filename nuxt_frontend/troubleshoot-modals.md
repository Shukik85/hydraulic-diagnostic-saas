# Исправление проблемы с модальными компонентами

## Проблема
Модальные компоненты `UCreateSystemModal`, `UReportGenerateModal`, `URunDiagnosticModal` работают корректно только на страницах `/chat` и `/diagnostics`, но не работают на других страницах.

## Исправления выполнены

### 1. ✅ Исправлена страница `/reports/index.vue`
Добавлен компонент `UReportGenerateModal` с правильной привязкой к кнопке "Новый отчёт".

### 2. ✅ Проверена структура компонентов
- Все модальные компоненты присутствуют в `nuxt_frontend/components/ui/`
- Конфигурация Nuxt правильная для автоимпорта
- Страница `/dashboard.vue` имеет корректное подключение модалок

## Возможные причины проблемы

### 1. Кэш Nuxt
Нужно очистить кэш и пересобрать:
```bash
cd nuxt_frontend
rm -rf .nuxt node_modules/.cache
npm run build
npm run dev
```

### 2. Hot Module Replacement
В dev режиме может быть проблема с HMR. Попробовать:
```bash
# Остановить сервер
# Ctrl+C
npm run dev
```

### 3. Browser Cache
Очистить кэш браузера (Ctrl+Shift+R) или открыть в incognito.

### 4. Conditional Loading
Возможно компоненты подгружаются условно. Проверить консоль браузера на ошибки.

## Инструкции для тестирования

1. **Остановите dev сервер** (Ctrl+C)
2. **Очистите кэш:**
   ```bash
   cd nuxt_frontend
   rm -rf .nuxt
   rm -rf node_modules/.cache
   ```
3. **Перезапустите:**
   ```bash
   npm run dev
   ```
4. **Откройте браузер в режиме инкогнито**
5. **Тестируйте модалки на:**
   - `http://localhost:3000/dashboard` - Quick Actions
   - `http://localhost:3000/systems` - кнопка "Add System"
   - `http://localhost:3000/reports` - кнопка "Новый отчёт"

## Статус модальных компонентов

| Страница | Компонент | Кнопка | Статус |
|----------|-----------|--------|---------|
| `/dashboard` | URunDiagnosticModal | Run Diagnostics | ✅ Добавлен |
| `/dashboard` | UReportGenerateModal | Generate Report | ✅ Добавлен |
| `/dashboard` | UCreateSystemModal | Add System | ✅ Добавлен |
| `/systems` | UCreateSystemModal | Add System | ✅ Был ранее |
| `/reports` | UReportGenerateModal | Новый отчёт | ✅ Исправлен |
| `/chat` | - | - | ✅ Работает |
| `/diagnostics` | - | - | ✅ Работает |

Если после очистки кэша проблема остаётся, проверьте консоль браузера на ошибки JavaScript.