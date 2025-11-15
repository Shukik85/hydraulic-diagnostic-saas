// composables/useMockData.ts
/**
 * Mock data для demo/testing.
 * Используется только когда backend недоступен или для презентаций.
 */
import { ref, computed } from 'vue'

export interface DiagnosticResult {
  id: number
  name: string
  equipment: string
  score: number
  issuesFound: number
  completedAt: string
  status: 'completed' | 'warning' | 'error' | 'processing'
  duration: string
}

export interface ActiveSession {
  id: number
  name: string
  equipment: string
  progress: number
  startedAt: string
}

/**
 * Composable для работы с mock данными.
 */
export function useMockData() {
  const config = useRuntimeConfig()
  
  // Feature flag check
  const isMockEnabled = computed(() => {
    return config.public.features?.enableMockData === true
  })
  
  /**
   * Mock результаты диагностики
   */
  const mockDiagnosticResults: DiagnosticResult[] = [
    {
      id: 1,
      name: 'Полный анализ системы - HYD-001',
      equipment: 'HYD-001 - Насосная станция A',
      score: 92,
      issuesFound: 1,
      completedAt: '2 часа назад',
      status: 'completed',
      duration: '4.2 мин'
    },
    {
      id: 2,
      name: 'Проверка давления - HYD-002',
      equipment: 'HYD-002 - Гидромотор B',
      score: 78,
      issuesFound: 3,
      completedAt: '6 часов назад',
      status: 'warning',
      duration: '2.8 мин'
    },
    {
      id: 3,
      name: 'Анализ температуры - HYD-003',
      equipment: 'HYD-003 - Регулирующий клапан C',
      score: 95,
      issuesFound: 0,
      completedAt: '1 день назад',
      status: 'completed',
      duration: '3.1 мин'
    },
    {
      id: 4,
      name: 'Критическая проверка - HYD-004',
      equipment: 'HYD-004 - Аккумулятор D',
      score: 45,
      issuesFound: 7,
      completedAt: '3 дня назад',
      status: 'error',
      duration: '5.7 мин'
    }
  ]
  
  /**
   * Mock RAG интерпретация (для demo)
   */
  const mockRAGInterpretation = {
    reasoning: `Шаг 1: Анализирую GNN результаты...
Обнаружены аномалии в узлах #5 и #12 графа.

Шаг 2: Проверяю базу знаний...
Найдены схожие случаи в документах по обслуживанию насосов.

Шаг 3: Коррелирую с историческими данными...
Паттерн соответствует износу подшипников.

Шаг 4: Формирую рекомендации...
Необходимо плановое ТО в течение 48 часов.`,
    
    summary: 'Обнаружены признаки износа подшипников в насосной станции HYD-001. Рекомендуется срочное плановое обслуживание для предотвращения отказа.',
    
    analysis: `Детальный анализ GNN результатов показывает:

1. **Аномалии вибрации**: Увеличение амплитуды на +35% за последние 72 часа
2. **Температурный тренд**: Постепенный рост на 2-3°C на подшипниковом узле
3. **Графовая структура**: Сильная связь между датчиками вибрации и температуры

Причина: Износ подшипников вследствие продолжительной эксплуатации (более 8000 часов).

Риск: Средний. При отсутствии вмешательства может привести к отказу в течение 7-14 дней.`,
    
    recommendations: [
      'Заменить подшипники в течение 48 часов',
      'Проверить качество смазки и уровень масла',
      'Выполнить балансировку ротора',
      'Установить дополнительный датчик вибрации для мониторинга'
    ],
    
    confidence: 0.87,
    
    knowledgeUsed: [
      {
        id: 'kb_001',
        title: 'Руководство по обслуживанию насосных станций',
        content: 'При обнаружении повышенной вибрации необходимо проверить состояние подшипников...',
        score: 0.92,
        metadata: {
          source: 'Техническая документация',
          category: 'Обслуживание',
          tags: ['насосы', 'вибрация', 'подшипники']
        }
      },
      {
        id: 'kb_002',
        title: 'Диагностика подшипников: признаки износа',
        content: 'Основные признаки: увеличение амплитуды вибрации, рост температуры, шумы...',
        score: 0.88,
        metadata: {
          source: 'База знаний',
          category: 'Диагностика',
          tags: ['подшипники', 'диагностика']
        }
      }
    ],
    
    metadata: {
      model: 'DeepSeek-R1-70B',
      processingTime: 12350,
      tokensUsed: 2847
    }
  }
  
  /**
   * Генерировать mock diagnostic result.
   */
  const generateMockResult = (): DiagnosticResult => {
    const id = Date.now()
    const score = Math.floor(Math.random() * 30) + 70
    const issuesFound = Math.floor(Math.random() * 5)
    
    return {
      id,
      name: `Анализ системы - HYD-${String(id).slice(-3)}`,
      equipment: `HYD-${String(id).slice(-3)} - Оборудование`,
      score,
      issuesFound,
      completedAt: 'только что',
      status: score >= 90 ? 'completed' : score >= 70 ? 'warning' : 'error',
      duration: `${Math.floor(Math.random() * 3) + 2}.${Math.floor(Math.random() * 9)} мин`
    }
  }
  
  /**
   * Симулировать активную сессию диагностики.
   */
  const simulateActiveSession = (
    onProgress: (progress: number) => void,
    onComplete: (result: DiagnosticResult) => void
  ) => {
    const session: ActiveSession = {
      id: Date.now(),
      name: `Новая диагностика`,
      equipment: 'HYD-001',
      progress: 0,
      startedAt: 'сейчас'
    }
    
    const interval = setInterval(() => {
      session.progress += Math.random() * 20
      onProgress(session.progress)
      
      if (session.progress >= 100) {
        session.progress = 100
        clearInterval(interval)
        
        setTimeout(() => {
          const result = generateMockResult()
          onComplete(result)
        }, 1000)
      }
    }, 500)
    
    return () => clearInterval(interval) // Cleanup function
  }
  
  return {
    isMockEnabled: readonly(isMockEnabled),
    mockDiagnosticResults,
    mockRAGInterpretation,
    generateMockResult,
    simulateActiveSession
  }
}

/**
 * Helper: проверить, включены ли mock данные.
 */
export function isMockDataEnabled(): boolean {
  const config = useRuntimeConfig()
  return config.public.features?.enableMockData === true
}