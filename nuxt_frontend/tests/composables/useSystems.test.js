import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { useSystems } from '../../composables/useSystems'

// Mock fetch globally
global.fetch = vi.fn()

describe('useSystems', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    vi.clearAllMocks()
    fetch.mockClear()
  })

  it('should fetch systems successfully', async () => {
    // Arrange
    const mockSystems = [
      { id: 1, name: 'System 1', status: 'active' },
      { id: 2, name: 'System 2', status: 'inactive' }
    ]
    
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockSystems
    })

    // Act
    const { systems, loading, error, fetchSystems } = useSystems()
    await fetchSystems()

    // Assert
    expect(systems.value).toEqual(mockSystems)
    expect(loading.value).toBe(false)
    expect(error.value).toBeNull()
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('should handle fetch error', async () => {
    // Arrange
    fetch.mockRejectedValueOnce(new Error('Network error'))

    // Act
    const { systems, loading, error, fetchSystems } = useSystems()
    await fetchSystems()

    // Assert
    expect(systems.value).toEqual([])
    expect(loading.value).toBe(false)
    expect(error.value).toBeTruthy()
  })

  it('should set loading state during fetch', async () => {
    // Arrange
    fetch.mockImplementationOnce(() => 
      new Promise(resolve => setTimeout(() => resolve({
        ok: true,
        json: async () => []
      }), 100))
    )

    // Act
    const { loading, fetchSystems } = useSystems()
    const fetchPromise = fetchSystems()
    
    // Assert - loading should be true during fetch
    expect(loading.value).toBe(true)
    
    await fetchPromise
    expect(loading.value).toBe(false)
  })

  it('should initialize with empty systems array', () => {
    // Act
    const { systems, loading, error } = useSystems()

    // Assert
    expect(systems.value).toEqual([])
    expect(loading.value).toBe(false)
    expect(error.value).toBeNull()
  })

  it('should handle malformed API response', async () => {
    // Arrange
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({ message: 'Internal Server Error' })
    })

    // Act
    const { systems, error, fetchSystems } = useSystems()
    await fetchSystems()

    // Assert
    expect(systems.value).toEqual([])
    expect(error.value).toBeTruthy()
  })
})
