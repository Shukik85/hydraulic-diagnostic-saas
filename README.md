Текущий код (rag_service.py):

1. Кеширование эмбеддингов документов
    _cache_document_embedding() - кеширование с бессрочным TTL
    _get_cached_document_embedding() - получение закешированных эмбеддингов
    _invalidate_document_cache() - инвалидация при обновлении документов

2. Кеширование результатов поиска
    _cache_search_result() - кеширование на 1 час (TTL=3600)
    _get_cached_search_result() - проверка кеша перед поиском
    search_documents() - новый метод с кешированием

3. Кеширование ответов на частые вопросы
    _cache_faq_answer() - кеширование на 24 часа (TTL=86400)
    _get_cached_faq_answer() - проверка кеша перед генерацией ответа

4. Версионирование ключей
    _get_cache_key() - генерация ключей с версионированием (:v1)
    Все ключи следуют паттерну: rag:{system_id}:{type}:{identifier}:{version}

5. Мониторинг hit/miss rate
    CacheStats - класс для отслеживания статистики
    increment_hit()/increment_miss() - увеличение счётчиков
    get_stats()/get_hit_rate() - получение статистики
    get_cache_stats() - метод для получения статистики в RagAssistant

6. Стратегии инвалидации
    При обновлении документа вызывается _invalidate_document_cache()
    Кешируются только те данные, которые безопасно кешировать
    Используется версионирование для инвалидации при изменениях схемы

7. Безопасность и производительность
    Все кешируемые данные проходят валидацию
    Используются параметризованные ключи для предотвращения коллизий
    Эффективное использование Redis с правильными TTL

Теперь RAG-модуль эффективно использует кеширование для улучшения производительности и снижения нагрузки на API OpenAI.

Запуск Celery worker:

celery -A core worker --loglevel=info
Запуск Celery beat (для периодических задач):

celery -A core beat --loglevel=info
Пример использования API:

POST /api/rag_assistant/documents/ - создание документа, возвращает task_id
POST /api/rag_assistant/systems/{id}/index/ - асинхронная индексация
GET /api/rag_assistant/tasks/{task_id}/ - проверка статуса задачи
