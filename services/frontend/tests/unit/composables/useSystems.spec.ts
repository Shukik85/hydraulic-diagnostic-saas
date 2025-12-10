/**
 * useSystems Composable Tests
 * @module tests/unit/composables/useSystems.spec.ts
 * @description Comprehensive unit tests for systems management composable
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { defineComponent, ref } from 'vue'
import { useSystems } from '~/composables/useSystems'
import type { SystemSummary, SystemDetail } from '~/types/systems'

// Mock data
const mockSystemSummary: SystemSummary = {
  systemId: 'sys-001',
  equipmentId: 'EXC-001',
  equipmentName: 'Komatsu PC200-8',
  equipmentType: 'excavator',
  status: 'online',
  lastUpdateAt: new Date().toISOString(),
  componentsCount: 3,
  sensorsCount: 5,
  topologyVersion: '1.0.0',
}

const mockSystemDetail: SystemDetail = {
  ...mockSystemSummary,
  operatingHours: 8500,
  components: [],
  edges: [],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
}

describe('useSystems', () => {
  let wrapper: any

  const createComponent = () => {
    const TestComponent = defineComponent({
      template: '<div></div>',
      setup() {
        return useSystems()
      },
    })

    wrapper = mount(TestComponent, {
      global: {
        stubs: {
          NuxtApp: true,
        },
      },
    })

    return wrapper.vm
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('initial state', () => {
    it('should initialize with empty systems array', () => {
      const { systems } = useSystems()
      expect(systems.value).toEqual([])
    })

    it('should initialize with loading as false', () => {
      const { loading } = useSystems()
      expect(loading.value).toBe(false)
    })

    it('should initialize with no error', () => {
      const { error } = useSystems()
      expect(error.value).toBeNull()
    })

    it('should initialize with pagination defaults', () => {
      const { pagination } = useSystems()
      expect(pagination.value.page).toBe(1)
      expect(pagination.value.pageSize).toBe(20)
      expect(pagination.value.total).toBe(0)
      expect(pagination.value.hasMore).toBe(false)
    })
  })

  describe('computed properties', () => {
    it('hasError should be true when error is set', () => {
      const { error, hasError } = useSystems()
      error.value = 'Test error'
      expect(hasError.value).toBe(true)
    })

    it('isEmpty should be true when systems array is empty', () => {
      const { systems, isEmpty } = useSystems()
      expect(isEmpty.value).toBe(true)
      systems.value.push(mockSystemSummary)
      expect(isEmpty.value).toBe(false)
    })

    it('systemsCount should return correct count', () => {
      const { systems, systemsCount } = useSystems()
      expect(systemsCount.value).toBe(0)
      systems.value.push(mockSystemSummary)
      expect(systemsCount.value).toBe(1)
    })
  })

  describe('clearError method', () => {
    it('should clear the error state', () => {
      const { error, clearError } = useSystems()
      error.value = 'Test error'
      expect(error.value).not.toBeNull()
      clearError()
      expect(error.value).toBeNull()
    })
  })

  describe('clearCurrentSystem method', () => {
    it('should clear current system and sensors', () => {
      const { currentSystem, sensors, clearCurrentSystem } = useSystems()
      currentSystem.value = mockSystemDetail
      sensors.value = [
        {
          sensorId: 'sensor-1',
          componentId: 'comp-1',
          sensorType: 'pressure',
          lastValue: 100,
          unit: 'bar',
          status: 'ok',
          lastUpdateAt: new Date().toISOString(),
        },
      ]

      clearCurrentSystem()

      expect(currentSystem.value).toBeNull()
      expect(sensors.value).toEqual([])
    })
  })

  describe('state management', () => {
    it('should manage systems list state', () => {
      const { systems } = useSystems()
      systems.value = [mockSystemSummary]
      expect(systems.value).toHaveLength(1)
      expect(systems.value[0].systemId).toBe('sys-001')
    })

    it('should manage current system state', () => {
      const { currentSystem } = useSystems()
      currentSystem.value = mockSystemDetail
      expect(currentSystem.value).not.toBeNull()
      expect(currentSystem.value?.systemId).toBe('sys-001')
    })

    it('should manage loading state', () => {
      const { loading } = useSystems()
      loading.value = true
      expect(loading.value).toBe(true)
      loading.value = false
      expect(loading.value).toBe(false)
    })

    it('should manage error state', () => {
      const { error } = useSystems()
      const errorMsg = 'Test error message'
      error.value = errorMsg
      expect(error.value).toBe(errorMsg)
    })
  })

  describe('pagination', () => {
    it('should update pagination state', () => {
      const { pagination } = useSystems()
      pagination.value.page = 2
      pagination.value.total = 100
      pagination.value.hasMore = true

      expect(pagination.value.page).toBe(2)
      expect(pagination.value.total).toBe(100)
      expect(pagination.value.hasMore).toBe(true)
    })
  })

  describe('error handling', () => {
    it('hasError computed should react to error changes', () => {
      const { error, hasError } = useSystems()
      expect(hasError.value).toBe(false)
      error.value = 'New error'
      expect(hasError.value).toBe(true)
      error.value = null
      expect(hasError.value).toBe(false)
    })
  })
})
