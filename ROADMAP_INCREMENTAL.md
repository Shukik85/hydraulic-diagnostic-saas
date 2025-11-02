# Roadmap: Enterprise Hydraulic Diagnostic Platform (Working Software After Every Step)

**МИССИЯ:** Построить продакшн-готовую диагностическую платформу с 4 ML моделями, 100+ типами датчиков, real-time мониторингом и 99.9% SLA. Каждый шаг остаётся работоспособным end-to-end.

**ТЕХНИЧЕСКИЕ ЦЕЛИ:**
- 99.5% точность ML моделей, <1% False Positive Rate
- <100ms inference latency от данных до алерта  
- 500+ датчиков на систему, 20+ протоколов (Modbus, OPC UA, MQTT)
- 5 лет истории данных, 99.9% uptime SLA
- Полная RU/EN локализация, мобильная адаптация

---

## Этап 0. Enterprise Frontend Foundation (ТЕКУЩИЙ)
**Статус:** ✅ ЗАВЕРШЁН
- ✅ Nuxt 4 + Tailwind v4 с premium-tokens.css design system
- ✅ Единый layout: dashboard с мобильным меню, user dropdown
- ✅ Полная RU/EN локализация (dashboard, settings, системы)
- ✅ Каркасы nested routes: sensors, equipment страницы
**Результат:** UI готов для подключения к Enterprise API

## Этап 1. Real-time Data Ingestion & TimescaleDB (неделя 1)
**Зависимости:** Этап 0
- Backend: TimescaleDB hypertables, continuous aggregates (hourly/daily)
- Ingestion API: 20+ протоколов (Modbus TCP/RTU, OPC UA, MQTT, CoAP, 4-20mA)
- Data pipeline: валидация, quarantine, downsampling, retention policies (5 лет)
- Metrics: ingestion rate, data quality, protocol health
**Результат:** Система принимает данные от 100+ типов датчиков в реальном времени

## Этап 2. ML Inference Engine - 4 модели (неделя 2)
**Зависимости:** Этап 1
- **HELM (Hierarchical ELM):** semi-supervised для неизвестных аномалий (99.5% точность, <1% FPR)
- **XGBoost:** supervised для критичных компонентов (valve, accumulator)
- **RandomForest:** анализ cooler и pump (стабильнее на малых датасетах)
- **Adaptive Thresholding:** динамические пороги с сезонной адаптацией
- Celery pipeline: <100ms inference latency, идемпотентность, ретраи
**Результат:** Система выявляет аномалии с lead time 2-6 часов до отказа

## Этап 3. Real-time Dashboard & Gauges (неделя 3)
**Зависимости:** Этапы 1-2
- Real-time gauges: стрелочные приборы для каждого датчика (как механические)
- Time-series charts: тренды за 24h/7d/30d/1y с зумом
- Heatmaps: корреляции между датчиками
- Alert timeline: история всех алертов с severity
- Equipment status: GREEN/YELLOW/RED статусы компонентов
- WebSocket: live updates, graceful reconnect
**Результат:** Операторы видят состояние системы в реальном времени

## Этап 4. Predictive Analytics & RUL (неделя 4)
**Зависимости:** Этапы 2-3
- Predictive RUL (Remaining Useful Life): когда нужна замена компонента
- Anomaly patterns: типизированная история аномалий
- Cost calculator: оценка убытков при отказе
- Component models: точность по компонентам (cooler 99.8%, valve 99.8%, pump 99.6%)
- Lead time prediction: 2-6 часов до критического отказа
**Результат:** Система предсказывает отказы с экономической оценкой

## Этап 5. Advanced Reporting System (неделя 5)
**Зависимости:** Этапы 1-4
- 40+ типов отчётов: ежедневный, еженедельный, ежемесячный, квартальный
- Автоматическая отправка: в 06:00 ежедневно, понедельник еженедельно
- PDF export: графики, таблицы, метрики, брендирование
- ROI калькулятор: сэкономлено ₽, предотвращено отказов
- Custom widgets: операторы создают свои метрики
**Результат:** Полная аналитика с автоматическими отчётами для менеджмента

## Этап 6. Multi-protocol Integrations (неделя 6)
**Зависимости:** Этапы 1-3
- SCADA интеграция: Modbus TCP/RTU, OPC UA, Profinet, Ethernet IP
- ERP/CMMS: автосоздание maintenance tickets
- Notification channels: Slack, Telegram, Email HTML, SMS
- Webhook API: POST для custom интеграций
- Protocol health: мониторинг подключений, автопереподключение
**Результат:** Бесшовная интеграция с существующими промышленными системами

## Этап 7. Enterprise Security & Compliance (неделя 7)
**Зависимости:** все предыдущие
- Authentication: OAuth 2.0, SAML 2.0, MFA (2FA)
- Data protection: TLS 1.3, AES-256, GDPR/152-ФЗ compliance
- FSTEC certified: на Яндекс.Облако (российское облако)
- RBAC: admin/engineer/operator/viewer роли
- Audit logs: все действия + SOC 2 Type II
- Backup & DR: ежедневные копии, восстановление за 1 час
**Результат:** Enterprise-ready безопасность с российской сертификацией

## Этап 8. Advanced ML & Custom Models (неделя 8)
**Зависимости:** Этапы 2-4
- Model versioning: A/B тестирование моделей
- Custom training: адаптация под конкретное оборудование клиента
- Ensemble methods: комбинирование 4 моделей для максимальной точности
- Online learning: модели дообучаются на новых данных
- Model explainability: почему модель выдала алерт
**Результат:** Самообучающаяся система с объяснимым ИИ

## Этап 9. Mobile App & Advanced UI (неделя 9)
**Зависимости:** Этапы 3-5
- iOS/Android native приложения
- Offline режим: кэширование критичных данных
- Push notifications: критичные алерты на телефон
- QR scanning: быстрое подключение к оборудованию
- Voice commands: голосовые запросы к системе
**Результат:** Мобильный доступ для техников в поле

## Этап 10. Enterprise Scaling & Performance (неделя 10)
**Зависимости:** все предыдущие
- Multi-tenant: изоляция данных между клиентами
- Horizontal scaling: auto-scaling под нагрузкой
- Performance: <100ms API, 99.9% uptime, неограниченные пользователи
- Advanced analytics: GraphQL API, Power BI connector
- White-label: кастомизация под бренд клиента
**Результат:** Enterprise platform готовая к масштабированию на тысячи систем

---

## Технические спецификации по этапам

### Этап 1: Data Infrastructure
- **Sensors:** 100+ типов (давление, температура, вибрация, поток, влажность)
- **Protocols:** Modbus TCP/RTU, OPC UA, MQTT, CoAP, 4-20mA, Profinet, Ethernet IP
- **Frequency:** 10-60 сек (настраивается по критичности)
- **Storage:** TimescaleDB с compression + retention (5 лет)

### Этап 2: ML Models Performance
- **HELM:** 99.5% accuracy, <1% FPR, semi-supervised
- **XGBoost:** 99.8% accuracy для valve/accumulator
- **RandomForest:** 99.6% accuracy для pump/cooler  
- **Adaptive:** динамические пороги, сезонная адаптация
- **Latency:** <100ms от данных до алерта

### Этап 3-4: Visualization & Analytics
- **Dashboards:** real-time gauges, time-series, heatmaps, alert timeline
- **RUL Prediction:** remaining useful life для каждого компонента
- **Lead time:** 2-6 часов предупреждения до критического отказа
- **Cost calculator:** экономическая оценка отказов

### Этап 5: Reporting & Integration
- **Reports:** 40+ типов, автоматическая отправка, PDF export
- **Integrations:** Slack, Telegram, Email, SMS, Jira, ServiceNow
- **API:** REST v1.0, webhook поддержка
- **Custom widgets:** пользовательские метрики

## SLA и Production Readiness
- **Uptime:** 99.9% SLA с мониторингом
- **Users:** неограниченные одновременные пользователи
- **Capacity:** 500+ датчиков на систему
- **Data retention:** 5 лет с компрессией
- **Security:** SOC 2 Type II, FSTEC certified
- **Support:** 24/7 для критичных алертов

## Roadmap Timeline
**Q4 2025 (СЕЙЧАС):** Этапы 0-3 (UI готов, начинаем data pipeline + ML)
**Q1 2026:** Этапы 4-6 (analytics, reporting, основные интеграции)
**Q2 2026:** Этапы 7-8 (enterprise security, advanced ML)
**Q3 2026:** Этапы 9-10 (mobile, scaling, белая метка)

---

## Принципы инкремента (без изменений)
- После каждого этапа: проект работает end-to-end
- API контракты фиксированы (OpenAPI), изменения через версионирование
- Feature flags для рискованных фич
- Тесты + наблюдаемость для всех новых функций