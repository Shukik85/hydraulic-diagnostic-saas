/**
 * System Metadata (5 Levels)
 * REQUIRED for diagnosis to function
 * Wizard creates this data → Diagnosis consumes it
 */

export interface SystemMetadata {
  systemId: string;
  systemName: string;
  createdAt: Date;
  updatedAt: Date;
  completionLevel: 1 | 2 | 3 | 4 | 5;
  aiReadinessScore: number;
  
  // Level 1: P&ID Schema (REQUIRED for diagnosis)
  level1?: {
    schemaFormat: 'pdf' | 'svg' | 'png' | 'jpg';
    schemaUrl: string;
    schemaFileName: string;
    uploadedAt: Date;
  };
  
  // Level 2: Sensor Placement (defines what sensors exist)
  level2?: {
    sensors: Array<{
      id: string;
      name: string;
      type: 'pressure' | 'temperature' | 'flow' | 'vibration' | 'position';
      coordinates?: { x: number; y: number };
    }>;
  };
  
  // Level 3: Nominal Values (thresholds for anomaly detection)
  level3?: {
    nominalValues: Record<string, {
      min: number;
      max: number;
      nominal: number;
      unit: string;
    }>;
  };
  
  // Level 4: Operating Modes (context for diagnosis)
  level4?: {
    operatingModes: Array<{
      name: string;
      description?: string;
      expectedValues?: Record<string, number>;
    }>;
  };
  
  // Level 5: AI Configuration (enables ML-powered diagnosis)
  level5?: {
    aiReadinessScore: number;
    modelType?: 'gnn' | 'lstm' | 'transformer';
    trainingStatus?: 'pending' | 'in_progress' | 'completed' | 'failed';
    lastTrainedAt?: Date;
  };
}

export interface SaveMetadataRequest {
  systemId: string;
  systemName: string;
  level1?: SystemMetadata['level1'];
  level2?: SystemMetadata['level2'];
  level3?: SystemMetadata['level3'];
  level4?: SystemMetadata['level4'];
  level5?: SystemMetadata['level5'];
}
