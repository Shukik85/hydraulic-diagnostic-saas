# Nuxt Frontend для гидравлической диагностической системы

Фронтенд приложение на Nuxt 3 для системы диагностики гидравлических систем.

## Структура проекта

```
nuxt_frontend/
├── components/          # Vue компоненты
│   └── SystemsList.vue # Компонент списка систем
├── pages/              # Страницы приложения (авто-роутинг)
│   └── index.vue       # Главная страница
├── app.vue             # Корневой компонент
├── package.json        # Зависимости проекта
└── README.md          # Этот файл
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

## Технологии

- **Nuxt 3** - Vue.js фреймворк для SSR и SSG
- **Vue 3** - Progressive JavaScript Framework
- **TypeScript** (опционально) - Типизированный JavaScript

## TODO

- [ ] Подключение к Django backend API
- [ ] Настройка аутентификации
- [ ] Добавление компонентов для диагностики
- [ ] Настройка Tailwind CSS / другого UI фреймворка
- [ ] Настройка state management (Pinia)

## Ссылки

- [Документация Nuxt 3](https://nuxt.com/docs)
- [Документация Vue 3](https://vuejs.org/)
