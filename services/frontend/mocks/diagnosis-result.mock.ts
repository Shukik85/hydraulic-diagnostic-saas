/**
 * Mock data for Diagnosis Result
 * Based on DiagnosisResultResponse type specification
 */

import type { DiagnosisResultResponse } from '~/types/diagnosis'

/**
 * Complete diagnosis result with GNN and RAG interpretation
 * Use for development and testing
 */
export const mockDiagnosisResult: DiagnosisResultResponse = {
  session_id: 'diag_20251114_143501',
  status: 'completed',
  created_at: '2025-11-14T14:35:01Z',
  completed_at: '2025-11-14T14:35:08Z',

  gnn_result: {
    predictions: [
      {
        component: 'Насос HYD-001',
        fault_type: 'Утечка давления',
        probability: 0.87,
        severity: 'warning',
        anomaly_indicators: {
          pressure: 0.82,
          temperature: 0.23,
          vibration: null,
          flow_rate: 0.65
        }
      },
      {
        component: 'Клапан CV-203',
        fault_type: null,
        probability: 0.15,
        severity: 'normal',
        anomaly_indicators: {
          pressure: 0.12,
          temperature: 0.08,
          vibration: null,
          flow_rate: 0.18
        }
      }
    ],
    anomaly_score: 0.78,
    overall_severity: 'warning',
    model_version: 'gnn_v2.1.0',
    inference_time_ms: 1247,
    confidence: 0.85
  },

  rag_interpretation: {
    summary: 'Обнаружена утечка давления в насосе HYD-001. Рекомендуется проверка уплотнений и замена изношенных компонентов в течение 48 часов.',

    reasoning: [
      {
        step: 1,
        title: 'Анализ аномалии давления',
        description: 'Давление в системе упало на 15% за последние 2 часа, что указывает на постепенную утечку.',
        evidence: [
          'Давление: 120 bar → 102 bar (-15%)',
          'Температура в пределах нормы (45°C)',
          'Расход масла увеличился на 8%'
        ],
        conclusion: 'Высокая вероятность утечки через уплотнения насоса'
      },
      {
        step: 2,
        title: 'Корреляция с историческими данными',
        description: 'Аналогичные паттерны наблюдались за 30 дней до предыдущего отказа насоса.',
        evidence: [
          'Схожий профиль деградации в Oct 2024',
          'Время работы: 15420 часов (близко к регламентной замене)'
        ],
        conclusion: 'Требуется профилактическое обслуживание'
      },
      {
        step: 3,
        title: 'Оценка рисков',
        description: 'При текущей скорости деградации полный отказ возможен через 48-72 часа.',
        evidence: [
          'Скорость падения давления: 2 bar/час',
          'Критический уровень: 95 bar'
        ],
        conclusion: 'Требуется срочное вмешательство'
      }
    ],

    recommendations: [
      {
        priority: 'high',
        action: 'Проверить уплотнения насоса HYD-001',
        rationale: 'Основная причина утечки давления по данным GNN модели',
        estimated_time: '2 часа',
        requires_shutdown: true,
        parts_needed: ['Уплотнительный комплект K-4521', 'Масло гидравлическое ISO VG 46']
      },
      {
        priority: 'high',
        action: 'Заменить изношенные поршневые кольца',
        rationale: 'Износ по наработке (15420 часов при регламенте 15000)',
        estimated_time: '4 часа',
        requires_shutdown: true,
        parts_needed: ['Поршневые кольца PN-8834']
      },
      {
        priority: 'medium',
        action: 'Провести диагностику всей системы',
        rationale: 'Исключить вторичные проблемы в связанных компонентах',
        estimated_time: '1 час',
        requires_shutdown: false,
        parts_needed: null
      },
      {
        priority: 'low',
        action: 'Обновить график технического обслуживания',
        rationale: 'Предотвратить повторение ситуации',
        estimated_time: '15 минут',
        requires_shutdown: false,
        parts_needed: null
      }
    ],

    severity: 'warning',
    confidence: 0.82,
    prognosis: 'При выполнении рекомендаций в течение 48 часов — полное восстановление. При игнорировании — высокий риск полного отказа системы через 3-4 дня с возможным повреждением смежных компонентов.',
    
    metadata: {
      model_version: 'gpt-4-turbo-2024-04-09',
      processing_time_ms: 2340,
      tokens_used: 1247,
      temperature: 0.3,
      rag_sources: 8
    }
  },

  equipment_context: {
    id: 'hyd-001',
    name: 'HYD-001',
    type: 'pump',
    operating_hours: 15420,
    last_maintenance: '2024-10-15T10:00:00Z',
    maintenance_interval: 15000,
    location: 'Участок А, Линия 3'
  },

  telemetry_snapshot: {
    pressure: [120, 118, 115, 110, 105, 102],
    temperature: [42, 43, 45, 45, 44, 45],
    flow_rate: [85, 87, 90, 92, 91, 93],
    timestamps: [
      '2025-11-14T12:00:00Z',
      '2025-11-14T12:30:00Z',
      '2025-11-14T13:00:00Z',
      '2025-11-14T13:30:00Z',
      '2025-11-14T14:00:00Z',
      '2025-11-14T14:30:00Z'
    ]
  }
}

/**
 * Diagnosis result without RAG (fallback scenario)
 */
export const mockDiagnosisResultWithoutRAG: DiagnosisResultResponse = {
  ...mockDiagnosisResult,
  rag_interpretation: null
}

/**
 * Critical severity example
 */
export const mockCriticalDiagnosis: DiagnosisResultResponse = {
  session_id: 'diag_20251114_150000',
  status: 'completed',
  created_at: '2025-11-14T15:00:00Z',
  completed_at: '2025-11-14T15:00:12Z',

  gnn_result: {
    predictions: [
      {
        component: 'Насос HYD-001',
        fault_type: 'Критический перегрев',
        probability: 0.95,
        severity: 'critical',
        anomaly_indicators: {
          pressure: 0.91,
          temperature: 0.98,
          vibration: 0.87,
          flow_rate: 0.45
        }
      }
    ],
    anomaly_score: 0.94,
    overall_severity: 'critical',
    model_version: 'gnn_v2.1.0',
    inference_time_ms: 1350,
    confidence: 0.93
  },

  rag_interpretation: {
    summary: 'КРИТИЧЕСКАЯ СИТУАЦИЯ: Обнаружен перегрев насоса до 95°C при допустимых 80°C. Немедленно остановите систему!',
    
    reasoning: [
      {
        step: 1,
        title: 'Критический перегрев',
        description: 'Температура превысила безопасный предел на 15°C.',
        evidence: [
          'Температура: 95°C (норма: 80°C)',
          'Скорость роста: 5°C/час',
          'Давление нестабильно'
        ],
        conclusion: 'НЕМЕДЛЕННАЯ ОСТАНОВКА ТРЕБУЕТСЯ'
      }
    ],

    recommendations: [
      {
        priority: 'high',
        action: 'НЕМЕДЛЕННО ОСТАНОВИТЬ СИСТЕМУ',
        rationale: 'Риск катастрофического отказа и повреждения персонала',
        estimated_time: 'Немедленно',
        requires_shutdown: true,
        parts_needed: null
      }
    ],

    severity: 'critical',
    confidence: 0.93,
    prognosis: 'Без немедленной остановки — высокий риск полного разрушения насоса и возгорания масла в течение часа.',
    
    metadata: {
      model_version: 'gpt-4-turbo-2024-04-09',
      processing_time_ms: 1890,
      tokens_used: 856,
      temperature: 0.3,
      rag_sources: 5
    }
  },

  equipment_context: {
    id: 'hyd-001',
    name: 'HYD-001',
    type: 'pump',
    operating_hours: 15420,
    last_maintenance: '2024-10-15T10:00:00Z',
    maintenance_interval: 15000,
    location: 'Участок А, Линия 3'
  }
}

/**
 * Normal/healthy system example
 */
export const mockNormalDiagnosis: DiagnosisResultResponse = {
  session_id: 'diag_20251114_160000',
  status: 'completed',
  created_at: '2025-11-14T16:00:00Z',
  completed_at: '2025-11-14T16:00:05Z',

  gnn_result: {
    predictions: [
      {
        component: 'Насос HYD-002',
        fault_type: null,
        probability: 0.05,
        severity: 'normal',
        anomaly_indicators: {
          pressure: 0.03,
          temperature: 0.12,
          vibration: 0.08,
          flow_rate: 0.05
        }
      }
    ],
    anomaly_score: 0.08,
    overall_severity: 'normal',
    model_version: 'gnn_v2.1.0',
    inference_time_ms: 987,
    confidence: 0.91
  },

  rag_interpretation: {
    summary: 'Система работает в штатном режиме. Все параметры в пределах нормы. Рекомендуется продолжить мониторинг.',
    
    reasoning: [
      {
        step: 1,
        title: 'Проверка ключевых параметров',
        description: 'Все измеренные параметры соответствуют эксплуатационным нормам.',
        evidence: [
          'Давление: 115 bar (норма: 100-120 bar)',
          'Температура: 55°C (норма: 40-80°C)',
          'Вибрация: 3mm/s (норма: <8mm/s)'
        ],
        conclusion: 'Система работает нормально'
      }
    ],

    recommendations: [
      {
        priority: 'low',
        action: 'Продолжить штатный мониторинг',
        rationale: 'Система в хорошем состоянии',
        estimated_time: 'Постоянно',
        requires_shutdown: false,
        parts_needed: null
      }
    ],

    severity: 'normal',
    confidence: 0.91,
    prognosis: 'Система стабильна. При соблюдении регламента обслуживания — длительная безотказная работа.',
    
    metadata: {
      model_version: 'gpt-4-turbo-2024-04-09',
      processing_time_ms: 1560,
      tokens_used: 634,
      temperature: 0.3,
      rag_sources: 3
    }
  },

  equipment_context: {
    id: 'hyd-002',
    name: 'HYD-002',
    type: 'pump',
    operating_hours: 8420,
    last_maintenance: '2025-09-01T10:00:00Z',
    maintenance_interval: 15000,
    location: 'Участок B, Линия 1'
  }
}

/**
 * Export all mocks
 */
export const diagnosisMocks = {
  standard: mockDiagnosisResult,
  withoutRAG: mockDiagnosisResultWithoutRAG,
  critical: mockCriticalDiagnosis,
  normal: mockNormalDiagnosis
}
