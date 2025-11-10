# Production Plan: ML Service до 15 ноября

## 🎯 Critical Path to Go-Live

### Phase 1: ML Service Ready (7-8 ноября)

#### Immediate Tasks (после завершения XGBoost)

**1. Production Inference API (2-3 часа)**
```bash
# Проверить готовые модели
ls -la models/*.joblib | grep "ноя  6"

# Запустить inference
make serve

# Тест всех endpoints
make test-predict
make test-health
make test-metrics
```

**2. API Enhancement (1-2 часа)**
- Добавить `/predict/batch` для массовых предсказаний
- Добавить `/models/info` для метаданных моделей
- Улучшить error handling и validation

**3. Performance Optimization (1-2 часа)**
- Model caching в памяти
- Request/response compression
- Async request processing

### Phase 2: Backend Integration (8-9 ноября)

**4. DRF Endpoints (3-4 часа)**
```python
# backend/apps/diagnostics/views.py
@api_view(['POST'])
def predict_fault(request):
    # Валидация входных данных
    serializer = SensorDataSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, 400)
    
    # Вызов ML service
    ml_result = requests.post(
        f'{ML_SERVICE_URL}/predict',
        json=serializer.validated_data
    )
    
    # Сохранение результата
    DiagnosticResult.objects.create(
        system_id=request.data['system_id'],
        prediction=ml_result.json(),
        timestamp=timezone.now()
    )
    
    return Response(ml_result.json())
```

**5. Real-time WebSocket (2-3 часа)**
- Channel layers для real-time updates
- Alert notifications при критических неисправностях
- Dashboard updates через WebSocket

### Phase 3: Data Pipeline (9-10 ноября)

**6. TimescaleDB Integration (4-6 часов)**
```python
# Sensor data ingestion
class SensorDataIngestor:
    async def ingest_batch(self, sensor_readings):
        # Сохранение в TimescaleDB
        await self.save_to_timescale(sensor_readings)
        
        # Триггер ML predictions
        predictions = await self.ml_service.predict(sensor_readings)
        
        # WebSocket notifications
        await self.notify_subscribers(predictions)
```

**7. Modbus/OPC UA MVP (6-8 часов)**
- Базовый Modbus TCP client
- OPC UA connection handling
- Data validation и quarantine

### Phase 4: Frontend Integration (10-11 ноября)

**8. Nuxt 4 Dashboard (4-6 часов)**
```vue
<!-- Real-time diagnostic dashboard -->
<template>
  <div class="diagnostic-dashboard">
    <SystemOverview :systems="systems" />
    <RealTimeAlerts :alerts="realtimeAlerts" />
    <SensorCharts :data="sensorData" />
    <MLPredictions :predictions="predictions" />
  </div>
</template>
```

**9. WebSocket Client (2-3 часа)**
- Real-time sensor data updates
- Live alert notifications
- Chart updates в реальном времени

### Phase 5: E2E Testing & Optimization (11-14 ноября)

**10. Performance Testing**
- Load testing API endpoints
- Latency optimization (<50ms p95)
- Memory usage optimization

**11. Security Hardening**
- API authentication/authorization
- Input validation
- Rate limiting

**12. Monitoring & Observability**
- Prometheus metrics
- Health checks
- Log aggregation

### Phase 6: Go-Live Preparation (14-15 ноября)

**13. Production Deployment**
- K8s manifests
- CI/CD pipeline
- Rollback procedures

**14. Documentation**
- API documentation
- Deployment guides
- Troubleshooting runbooks

## 🚀 Success Metrics

- **API Latency:** <50ms p95
- **Uptime:** 99.99%
- **ML Accuracy:** >99% (уже достигнуто)
- **Real-time processing:** <100ms sensor to alert
- **Scalability:** Handle 1000+ concurrent requests

## 📊 Current Status

- ✅ ML Models training (XGBoost в процессе)
- ✅ Docker containerization ready
- ✅ FastAPI inference API базовый
- 🔄 Production optimization needed
- ❌ Backend integration pending
- ❌ Frontend dashboard pending
- ❌ Data ingestion pipeline pending

## 🎯 Daily Milestones

**7 ноября:** ML Service production-ready
**8 ноября:** Backend API integration
**9 ноября:** Data pipeline MVP
**10 ноября:** Frontend dashboard
**11 ноября:** E2E testing
**12-13 ноября:** Performance optimization
**14 ноября:** Production deployment
**15 ноября утро:** Go-live!
