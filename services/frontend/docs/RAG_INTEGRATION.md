# 🤖 RAG Integration Guide

> Интеграция RAG (Retrieval-Augmented Generation) в frontend

**RAG Service:** DeepSeek-R1 (70B parameters)  
**Vector DB:** FAISS with E5-multilingual embeddings  
**API:** FastAPI with async endpoints

---

## 🎯 What is RAG?

**RAG = Retrieval-Augmented Generation**

AI модель, которая:
1. ✅ **Retrieves** - находит релевантные документы из Knowledge Base
2. ✅ **Augments** - добавляет контекст в prompt
3. ✅ **Generates** - генерирует ответ с учетом контекста

**Для нашей платформы:**
- 📊 **GNN Results** → structured anomaly data
- 📚 **Knowledge Base** → техническая документация, история ремонтов
- 🤖 **DeepSeek-R1** → интерпретация с reasoning

**Result:**  
Не просто "обнаружена аномалия", а **"почему, что делать, когда и какие риски"**

---

## 🚀 Quick Start

### 1. Enable RAG Feature

```bash
# .env
NUXT_PUBLIC_ENABLE_RAG=true
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
```

### 2. Use in Component

```vue
<script setup lang="ts">
import { useRAG } from '~/composables/useRAG'

const { interpretDiagnosis, loading, error } = useRAG()

const handleInterpret = async () => {
  const interpretation = await interpretDiagnosis({
    gnnResults: diagnosticData,
    equipmentId: 'exc_001',
    useKnowledgeBase: true
  })
  
  console.log('AI Interpretation:', interpretation)
}
</script>

<template>
  <div>
    <button @click="handleInterpret" :disabled="loading">
      Генерировать интерпретацию
    </button>
    
    <RagInterpretationPanel
      :interpretation="interpretation"
      :loading="loading"
      :error="error"
    />
  </div>
</template>
```

---

## 📚 API Reference

### `useRAG()` Composable

#### Methods

##### `interpretDiagnosis(request: RAGInterpretationRequest)`

Интерпретировать GNN результаты через DeepSeek-R1.

**Parameters:**
```typescript
interface RAGInterpretationRequest {
  gnnResults: any              // GNN output
  equipmentId: string          // Equipment ID
  equipmentContext?: {...}     // Опциональный контекст
  useKnowledgeBase?: boolean   // Использовать KB (default: true)
}
```

**Returns:**
```typescript
interface RAGInterpretationResponse {
  reasoning: string           // Процесс рассуждения
  summary: string            // Краткая сводка
  analysis: string           // Детальный анализ
  recommendations: string[]  // Рекомендации
  confidence: number         // 0-1 (уверенность)
  knowledgeUsed: [...]       // Использованные документы
  metadata: {...}            // Метаданные (время, tokens, model)
}
```

**Example:**
```typescript
const { interpretDiagnosis, loading } = useRAG()

const interpretation = await interpretDiagnosis({
  gnnResults: {
    anomalies: [
      { nodeId: 5, score: 0.87, type: 'vibration' },
      { nodeId: 12, score: 0.65, type: 'temperature' }
    ],
    graphStructure: {...}
  },
  equipmentId: 'exc_001',
  equipmentContext: {
    name: 'Насосная станция A',
    type: 'hydraulic_pump',
    operatingHours: 8342
  },
  useKnowledgeBase: true
})

console.log(interpretation.summary)  // "Обнаружены признаки..."
```

---

##### `searchKnowledgeBase(query, topK)`

Поиск документов в Knowledge Base.

**Parameters:**
```typescript
query: string     // Поисковый запрос
 topK: number      // Количество результатов (default: 5)
```

**Returns:**
```typescript
interface KnowledgeBaseSearchResponse {
  documents: KnowledgeDocument[]
  totalResults: number
  searchTime: number  // ms
}
```

**Example:**
```typescript
const { searchKnowledgeBase } = useRAG()

const results = await searchKnowledgeBase('износ подшипников', 3)

results.documents.forEach(doc => {
  console.log(doc.title, doc.score)  // 0.92
})
```

---

##### `explainAnomaly(anomalyData)`

Быстрое объяснение аномалии.

**Parameters:**
```typescript
anomalyData: any  // Данные об аномалии
```

**Returns:**
```typescript
string | null  // Объяснение (краткое)
```

**Example:**
```typescript
const explanation = await explainAnomaly({
  type: 'vibration',
  value: 2.5,
  threshold: 1.8,
  timestamp: '2025-11-15T09:00:00Z'
})

console.log(explanation)
// "Повышенная вибрация может указывать на износ подшипников..."
```

---

## 🎨 UI Components

### InterpretationPanel

Основной UI component для отображения RAG интерпретаций.

**Props:**
```typescript
interface Props {
  interpretation: RAGInterpretationResponse | null
  loading?: boolean
  error?: Error | null
}
```

**Events:**
```typescript
emit('retry')      // Повтор генерации
emit('generate')   // Запустить генерацию
```

**Usage:**
```vue
<template>
  <RagInterpretationPanel
    :interpretation="interpretation"
    :loading="loading"
    :error="error"
    @retry="handleRetry"
    @generate="handleGenerate"
  />
</template>
```

**Features:**
- ✅ Summary card (краткая сводка)
- ✅ Reasoning process (collapsible)
- ✅ Detailed analysis
- ✅ Recommendations list
- ✅ Knowledge base context
- ✅ Confidence indicator
- ✅ Loading/error states

---

### ReasoningSteps Component

Визуализация процесса рассуждения AI.

**Props:**
```typescript
interface Props {
  reasoning: string  // Raw reasoning text
}
```

**Example:**
```vue
<RagReasoningSteps :reasoning="interpretation.reasoning" />
```

**Output:**
```
🧠 Процесс рассуждения:

✅ Шаг 1: Анализирую GNN результаты...
✅ Шаг 2: Проверяю базу знаний...
✅ Шаг 3: Коррелирую с историей...
✅ Шаг 4: Формирую рекомендации...
```

---

### KnowledgeContext Component

Показывает использованные документы из KB.

**Props:**
```typescript
interface Props {
  documents: KnowledgeDocument[]
}
```

**Example:**
```vue
<RagKnowledgeContext :documents="interpretation.knowledgeUsed" />
```

---

## 📊 Integration Patterns

### Pattern 1: Diagnostic Page + RAG

**Use Case:** Полная страница диагностики с interpretation

```vue
<script setup lang="ts">
const route = useRoute()
const api = useGeneratedApi()
const { interpretDiagnosis, loading: ragLoading } = useRAG()

// 1. Load diagnostic result
const { data: diagnostic, pending } = await useAsyncData(
  `diagnostic-${route.params.id}`,
  () => api.diagnosis.getDiagnosticResult(route.params.id)
)

// 2. Generate interpretation
const interpretation = ref(null)

const handleGenerate = async () => {
  if (!diagnostic.value) return
  
  interpretation.value = await interpretDiagnosis({
    gnnResults: diagnostic.value.gnnOutput,
    equipmentId: diagnostic.value.equipmentId,
    useKnowledgeBase: true
  })
}

// Auto-generate on load
onMounted(() => {
  if (diagnostic.value) {
    handleGenerate()
  }
})
</script>

<template>
  <div>
    <!-- Diagnostic Results -->
    <DiagnosticResultsCard :data="diagnostic" :loading="pending" />
    
    <!-- RAG Interpretation -->
    <RagInterpretationPanel
      :interpretation="interpretation"
      :loading="ragLoading"
      @generate="handleGenerate"
    />
  </div>
</template>
```

---

### Pattern 2: Inline Explanation

**Use Case:** Быстрое объяснение аномалии

```vue
<script setup lang="ts">
const { explainAnomaly, loading } = useRAG()

const props = defineProps<{
  anomaly: any
}>()

const explanation = ref('')

const showExplanation = async () => {
  explanation.value = await explainAnomaly(props.anomaly) || 'Нет объяснения'
}
</script>

<template>
  <div class="anomaly-card">
    <h4>{{ anomaly.type }}</h4>
    <p>{{ anomaly.value }}</p>
    
    <!-- Show explanation -->
    <button @click="showExplanation" :disabled="loading">
      🤖 Объяснить
    </button>
    
    <p v-if="explanation" class="text-sm text-gray-600 mt-2">
      {{ explanation }}
    </p>
  </div>
</template>
```

---

### Pattern 3: Knowledge Base Search

**Use Case:** Поиск в технической документации

```vue
<script setup lang="ts">
const { searchKnowledgeBase, loading } = useRAG()

const query = ref('')
const results = ref([])

const handleSearch = async () => {
  const response = await searchKnowledgeBase(query.value, 10)
  results.value = response?.documents || []
}
</script>

<template>
  <div>
    <input v-model="query" @keyup.enter="handleSearch" />
    
    <div v-for="doc in results" :key="doc.id">
      <h4>{{ doc.title }}</h4>
      <p>{{ doc.content }}</p>
      <span class="badge">{{ Math.round(doc.score * 100) }}% match</span>
    </div>
  </div>
</template>
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Enable RAG feature
NUXT_PUBLIC_ENABLE_RAG=true

# API endpoint
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1

# Timeout (ms)
NUXT_PUBLIC_API_TIMEOUT=30000
```

### Runtime Config

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  runtimeConfig: {
    public: {
      features: {
        ragInterpretation: process.env.ENABLE_RAG === 'true'
      }
    }
  }
})
```

### Feature Flag Check

```typescript
const config = useRuntimeConfig()

if (config.public.features.ragInterpretation) {
  // RAG enabled
} else {
  // Fallback to basic mode
}
```

---

## 💡 Best Practices

### 1. Always Check Feature Flag

```typescript
// ✅ Good
const { isRAGEnabled } = useRAG()

if (isRAGEnabled.value) {
  await interpretDiagnosis({...})
} else {
  console.log('RAG disabled, using basic mode')
}

// ❌ Bad
await interpretDiagnosis({...})  // Может упасть!
```

### 2. Handle Errors Gracefully

```typescript
// ✅ Good
try {
  const interpretation = await interpretDiagnosis({...})
  if (!interpretation) {
    showBasicMode()
  }
} catch (error) {
  console.error('RAG error:', error)
  showFallbackUI()
}
```

### 3. Show Loading States

```vue
<!-- ✅ Good -->
<div v-if="loading">
  <Spinner /> Генерирую интерпретацию...
</div>

<!-- ❌ Bad -->
<div>
  {{ interpretation }}  <!-- Пусто во время loading -->
</div>
```

### 4. Cache Interpretations

```typescript
// Cache в localStorage
const cacheKey = `rag_${diagnosticId}`
const cached = localStorage.getItem(cacheKey)

if (cached) {
  interpretation.value = JSON.parse(cached)
} else {
  interpretation.value = await interpretDiagnosis({...})
  localStorage.setItem(cacheKey, JSON.stringify(interpretation.value))
}
```

---

## 🐛 Troubleshooting

### RAG Service Not Available

**Symptom:**
```
Error: Failed to fetch RAG interpretation
```

**Check:**
```bash
# 1. RAG Service running?
curl http://localhost:8004/health

# 2. Environment variable set?
echo $NUXT_PUBLIC_ENABLE_RAG

# 3. Feature flag enabled?
const { isRAGEnabled } = useRAG()
console.log(isRAGEnabled.value)  // should be true
```

**Fix:**
```bash
# Start RAG service
cd services/rag
docker-compose up -d

# Enable feature
NUXT_PUBLIC_ENABLE_RAG=true npm run dev
```

---

### Slow Response Times

**Symptom:**
RAG interpretation занимает > 30 секунд.

**Причины:**
1. **Large GNN results** - слишком много данных
2. **Knowledge Base search** - много документов
3. **Model loading** - первый запрос

**Fix:**
```typescript
// Reduce max_tokens
const { interpretDiagnosis } = useRAG({ maxTokens: 1024 })

// Or limit KB search
const { searchKnowledgeBase } = useRAG()
await searchKnowledgeBase(query, 3)  // topK = 3 instead of 5
```

---

### Knowledge Base Empty

**Symptom:**
`knowledgeUsed: []` - нет документов

**Check:**
```bash
# KB has documents?
curl http://localhost:8004/kb/stats

# Response should show:
# {
#   "total_documents": 50,
#   "total_vectors": 250
# }
```

**Fix:**
```bash
# Populate Knowledge Base
curl -X POST http://localhost:8004/kb/add \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Руководство по обслуживанию",
    "content": "..."
  }'
```

---

## 📈 Performance Optimization

### 1. Debounce KB Search

```typescript
import { useDebounceFn } from '@vueuse/core'

const debouncedSearch = useDebounceFn(async (query: string) => {
  const results = await searchKnowledgeBase(query)
  // ...
}, 300)  // 300ms debounce

watch(searchQuery, (newQuery) => {
  debouncedSearch(newQuery)
})
```

### 2. Cache Interpretations

```typescript
const interpretationCache = new Map<string, RAGInterpretationResponse>()

const getCachedInterpretation = async (diagnosticId: string) => {
  if (interpretationCache.has(diagnosticId)) {
    return interpretationCache.get(diagnosticId)
  }
  
  const result = await interpretDiagnosis({...})
  interpretationCache.set(diagnosticId, result)
  return result
}
```

### 3. Progressive Loading

```typescript
// Show summary first
const interpretation = ref(null)

// Step 1: Show summary immediately
interpretation.value = { summary: 'Loading...' }

// Step 2: Load full interpretation
const full = await interpretDiagnosis({...})
interpretation.value = full
```

---

## 📖 Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Общая архитектура
- [API_INTEGRATION.md](./API_INTEGRATION.md) - API integration guide
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Production deployment

---

**Last Updated:** November 15, 2025  
**Author:** Plotnikov Aleksandr  
**Contact:** shukik85@ya.ru