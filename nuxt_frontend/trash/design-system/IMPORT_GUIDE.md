# Инструкция по импорту в Figma Desktop App

## SVG Макеты (key-screens/)

1. Откройте Figma Desktop
2. Создайте новый файл или откройте существующий
3. Перетащите SVG файлы (`dashboard.svg`, `diagnostics.svg`) из проводника в рабочую область Figma
4. SVG автоматически конвертируется в векторные фреймы

## JSON Компоненты (components/)

**Основной способ для Desktop App:**

1. Откройте `.fig` файл в любом текстовом редакторе (VS Code, Notepad++)
2. Скопируйте все содержимое JSON файла (Ctrl+A, Ctrl+C)
3. В Figma Desktop: Edit → Paste JSON (или Ctrl+Shift+V)
4. Компонент будет создан в текущем файле как новый элемент

**Альтернативный способ:**

1. Создайте новый пустой файл в Figma
2. Перетащите `.fig` файл прямо в рабочую область
3. Figma попытается импортировать как JSON

## Темы (themes/)

**Импорт цветовых стилей:**

1. Откройте файл темы (light.fig или dark.fig) в текстовом редакторе
2. Скопируйте JSON содержимое
3. В Figma: Edit → Paste JSON
4. Создайте локальные стили вручную:
   - Правой кнопкой на элементе → "Create style" → "Color style"
   - Для типографики: "Create style" → "Text style"
   - Для эффектов: "Create style" → "Effect style"

**Быстрая настройка темы:**

1. Создайте новый файл для дизайн-системы
2. Используйте значения из JSON файлов для создания стилей:
   - Colors: `#0066CC` для primary, `#FFFFFF` для background и т.д.
   - Typography: Inter Bold 24px для заголовков
   - Spacing: 16px для основных отступов

## Практические шаги для быстрого старта

### Шаг 1: Импорт макетов

- Перетащите `dashboard.svg` в новый файл Figma
- Перетащите `diagnostics.svg` в тот же файл

### Шаг 2: Создание компонентов

- Откройте `button.fig` в VS Code
- Скопируйте JSON
- В Figma: Ctrl+Shift+V (Paste JSON)
- Повторите для `card.fig` и `chart.fig`

### Шаг 3: Настройка цветов

- Создайте Color Styles в панели справа
- Добавьте цвета из light.fig:
  - Primary: #0066CC
  - Secondary: #FF9900
  - Success: #388E3C
  - Background: #FFFFFF
  - Text: #212121

## Структура файлов

```
design-system/
├── README.md              # Документация системы
├── IMPORT_GUIDE.md        # Эта инструкция
├── key-screens/          # SVG макеты
│   ├── dashboard.svg      # Дашборд с метриками
│   └── diagnostics.svg    # Диагностика с графиками
├── components/           # JSON компоненты
│   ├── button.fig         # Кнопки (Primary/Secondary/Warning)
│   ├── card.fig          # Карточки (Default/Elevated/Outlined)
│   └── chart.fig         # Графики (Line/Bar/Gauge)
└── themes/              # Темы
    ├── light.fig         # Светлая тема (WCAG AAA)
    └── dark.fig          # Темная тема
```

## Troubleshooting для Figma Desktop

- **Paste JSON не работает**: Убедитесь, что JSON валиден (без лишних символов)
- **Файлы не импортируются**: Проверьте, что файлы не повреждены
- **Стили не применяются**: Создайте стили вручную через правую панель
- **Компоненты не отображаются**: Попробуйте Paste JSON в пустом файле

## Советы по работе

- Создайте отдельный файл "Design System" для всех компонентов
- Используйте Local Styles для консистентности
- Для тем используйте Figma Variables (если доступно в вашей версии)
- Регулярно обновляйте компоненты в Team Library
