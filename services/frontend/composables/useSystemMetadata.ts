/**
 * System Metadata Management
 * 
 * Wizard creates metadata → Diagnosis uses metadata
 * Without metadata, diagnosis cannot interpret sensor data
 */

import type { SystemMetadata, SaveMetadataRequest } from '~/types/metadata';

export function useSystemMetadata() {
  const toast = useToast();
  
  const metadata = ref<SystemMetadata | null>(null);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  /**
   * Save metadata (Wizard → Backend)
   */
  const saveMetadata = async (data: SaveMetadataRequest): Promise<boolean> => {
    isLoading.value = true;
    error.value = null;

    try {
      // Production: POST /api/v1/systems/{systemId}/metadata
      // This stores: schema, sensors, thresholds, modes, AI config
      
      const metadataToSave: SystemMetadata = {
        systemId: data.systemId,
        systemName: data.systemName,
        createdAt: metadata.value?.createdAt || new Date(),
        updatedAt: new Date(),
        completionLevel: calculateCompletionLevel(data),
        aiReadinessScore: calculateReadinessScore(data),
        level1: data.level1,
        level2: data.level2,
        level3: data.level3,
        level4: data.level4,
        level5: data.level5,
      };

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Temporary: localStorage (production uses database)
      localStorage.setItem(
        `system-metadata-${data.systemId}`,
        JSON.stringify(metadataToSave)
      );

      metadata.value = metadataToSave;
      toast.success('System metadata saved', 'Ready for diagnosis');
      return true;
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to save metadata';
      toast.error(error.value, 'Save Error');
      return false;
    } finally {
      isLoading.value = false;
    }
  };

  /**
   * Load metadata (Diagnosis ← Backend)
   * REQUIRED before showing diagnosis page
   */
  const loadMetadata = async (systemId: string): Promise<SystemMetadata | null> => {
    isLoading.value = true;
    error.value = null;

    try {
      // Production: GET /api/v1/systems/{systemId}/metadata
      // This returns: schema, sensors, thresholds, modes, AI config
      
      await new Promise(resolve => setTimeout(resolve, 300));
      
      const stored = localStorage.getItem(`system-metadata-${systemId}`);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Restore Date objects
        parsed.createdAt = new Date(parsed.createdAt);
        parsed.updatedAt = new Date(parsed.updatedAt);
        if (parsed.level1?.uploadedAt) {
          parsed.level1.uploadedAt = new Date(parsed.level1.uploadedAt);
        }
        
        metadata.value = parsed;
        return parsed;
      }
      
      return null;
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to load metadata';
      return null;
    } finally {
      isLoading.value = false;
    }
  };

  /**
   * Check if system is ready for diagnosis
   */
  const isReadyForDiagnosis = (systemId: string): boolean => {
    const stored = localStorage.getItem(`system-metadata-${systemId}`);
    if (!stored) return false;
    
    try {
      const meta: SystemMetadata = JSON.parse(stored);
      // Minimum requirements: Level 1 (schema) + Level 2 (sensors)
      return !!(meta.level1 && meta.level2 && meta.level2.sensors.length > 0);
    } catch {
      return false;
    }
  };

  /**
   * Get sensor list from metadata
   */
  const getSensors = (systemId: string): Array<any> => {
    const stored = localStorage.getItem(`system-metadata-${systemId}`);
    if (!stored) return [];
    
    try {
      const meta: SystemMetadata = JSON.parse(stored);
      return meta.level2?.sensors || [];
    } catch {
      return [];
    }
  };

  /**
   * Get nominal values from metadata
   */
  const getNominalValues = (systemId: string): Record<string, any> => {
    const stored = localStorage.getItem(`system-metadata-${systemId}`);
    if (!stored) return {};
    
    try {
      const meta: SystemMetadata = JSON.parse(stored);
      return meta.level3?.nominalValues || {};
    } catch {
      return {};
    }
  };

  return {
    metadata: readonly(metadata),
    isLoading: readonly(isLoading),
    error: readonly(error),
    saveMetadata,
    loadMetadata,
    isReadyForDiagnosis,
    getSensors,
    getNominalValues,
  };
}

/**
 * Calculate completion level (1-5)
 */
function calculateCompletionLevel(data: Partial<SystemMetadata>): 1 | 2 | 3 | 4 | 5 {
  if (data.level5) return 5;
  if (data.level4) return 4;
  if (data.level3) return 3;
  if (data.level2) return 2;
  return 1;
}

/**
 * Calculate AI readiness score (0-100)
 */
function calculateReadinessScore(data: Partial<SystemMetadata>): number {
  let score = 0;
  
  if (data.level1?.schemaUrl) score += 20;
  if (data.level2?.sensors) score += Math.min(data.level2.sensors.length * 10, 30);
  if (data.level3?.nominalValues && Object.keys(data.level3.nominalValues).length > 0) score += 25;
  if (data.level4?.operatingModes) score += Math.min(data.level4.operatingModes.length * 5, 25);
  
  return Math.min(score, 100);
}
