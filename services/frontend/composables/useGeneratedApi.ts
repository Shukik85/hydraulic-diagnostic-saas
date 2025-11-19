// services/frontend/composables/useGeneratedApi.ts
/**
 * Wrapper для auto-generated OpenAPI client.
 * Предоставляет typed API clients для всех backend сервисов.
 */
import {
  DiagnosisService,
  EquipmentService,
  GNNService,
  RAGService,
  Configuration
} from '~/generated/api/services'

import { OpenAPI } from '~/generated/api/core/OpenAPI'
import { useRuntimeConfig } from 'nuxt/app'

/**
 * Создает configured API clients для всех сервисов.
 */
// export const useGeneratedApi = () => {
//   const config = useRuntimeConfig()

//   // OpenAPI config доступен напрямую (новый openapi-typescript)
//   OpenAPI.BASE = config.public.apiBase
//   OpenAPI.TOKEN = undefined // TODO: добавить интеграцию с authStore

//   return {
//     diagnosis: new DiagnosisService(),
//     equipment: new EquipmentService(),
//     gnn: new GNNService(),
//     rag: new RAGService()
//   }
// }

// Временно!
export const useGeneratedApi = () => ({
  login: async () => undefined,
  register: async () => undefined,
  logout: async () => undefined,
  isAuthenticated: async () => true,
  getCurrentUser: async () => null,
  updateUser: async () => undefined,
  api: { request: async () => undefined }, // Для systems.store.ts (api.request)
  request: async () => undefined,
  // Сервисы, чтобы не ломать существующий код:
  diagnosis: {},
  equipment: {},
  gnn: {},
  rag: {},
});
