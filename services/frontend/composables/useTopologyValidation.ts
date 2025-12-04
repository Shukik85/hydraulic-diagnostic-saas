import type {
  GraphTopology,
  Component,
  Edge,
  EquipmentMetadata,
  ValidationError,
} from '~/types/gnn';

export const useTopologyValidation = () => {
  /**
   * Validate equipment metadata
   */
  const validateEquipment = (equipment: EquipmentMetadata): ValidationError[] => {
    const errors: ValidationError[] = [];

    // Equipment ID: 1-100 chars, alphanumeric
    if (!equipment.equipmentId) {
      errors.push({ field: 'equipmentId', message: 'Equipment ID is required' });
    } else if (equipment.equipmentId.length > 100) {
      errors.push({ field: 'equipmentId', message: 'Equipment ID must be ≤100 characters' });
    } else if (!/^[a-zA-Z0-9-_]+$/.test(equipment.equipmentId)) {
      errors.push({ field: 'equipmentId', message: 'Equipment ID must be alphanumeric' });
    }

    // Equipment Name: 1-255 chars
    if (!equipment.equipmentName) {
      errors.push({ field: 'equipmentName', message: 'Equipment name is required' });
    } else if (equipment.equipmentName.length > 255) {
      errors.push({ field: 'equipmentName', message: 'Equipment name must be ≤255 characters' });
    }

    // Operating hours: >= 0
    if (equipment.operatingHours !== undefined && equipment.operatingHours < 0) {
      errors.push({ field: 'operatingHours', message: 'Operating hours must be ≥0' });
    }

    return errors;
  };

  /**
   * Validate single component
   */
  const validateComponent = (component: Component, index: number): ValidationError[] => {
    const errors: ValidationError[] = [];
    const prefix = `components[${index}]`;

    // Component ID: 1-50 chars, alphanumeric
    if (!component.componentId) {
      errors.push({ field: `${prefix}.componentId`, message: 'Component ID is required' });
    } else if (component.componentId.length > 50) {
      errors.push({ field: `${prefix}.componentId`, message: 'Component ID must be ≤50 characters' });
    } else if (!/^[a-zA-Z0-9-_]+$/.test(component.componentId)) {
      errors.push({ field: `${prefix}.componentId`, message: 'Component ID must be alphanumeric' });
    }

    // Component type
    if (!component.componentType) {
      errors.push({ field: `${prefix}.componentType`, message: 'Component type is required' });
    }

    // Sensors: at least 1
    if (!component.sensors || component.sensors.length === 0) {
      errors.push({ field: `${prefix}.sensors`, message: 'At least 1 sensor is required' });
    }

    // Nominal pressure: 0-1000
    if (component.nominalPressureBar !== undefined) {
      if (component.nominalPressureBar < 0) {
        errors.push({ field: `${prefix}.nominalPressureBar`, message: 'Pressure must be ≥0' });
      } else if (component.nominalPressureBar > 1000) {
        errors.push({ field: `${prefix}.nominalPressureBar`, message: 'Pressure must be ≤1000 bar' });
      }
    }

    // Nominal flow: 0-1000
    if (component.nominalFlowLpm !== undefined) {
      if (component.nominalFlowLpm < 0) {
        errors.push({ field: `${prefix}.nominalFlowLpm`, message: 'Flow must be ≥0' });
      } else if (component.nominalFlowLpm > 1000) {
        errors.push({ field: `${prefix}.nominalFlowLpm`, message: 'Flow must be ≤1000 L/min' });
      }
    }

    // Rated power: >= 0
    if (component.ratedPowerKw !== undefined && component.ratedPowerKw < 0) {
      errors.push({ field: `${prefix}.ratedPowerKw`, message: 'Power must be ≥0' });
    }

    return errors;
  };

  /**
   * Validate components array
   */
  const validateComponents = (components: Component[]): ValidationError[] => {
    let errors: ValidationError[] = [];

    // Min 2 components required
    if (components.length < 2) {
      errors.push({ field: 'components', message: 'At least 2 components are required' });
      return errors;
    }

    // Validate each component
    components.forEach((component, index) => {
      errors = errors.concat(validateComponent(component, index));
    });

    // Check for duplicate component IDs
    const ids = components.map(c => c.componentId);
    const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
    if (duplicates.length > 0) {
      errors.push({
        field: 'components',
        message: `Duplicate component IDs: ${duplicates.join(', ')}`,
      });
    }

    return errors;
  };

  /**
   * Validate single edge
   */
  const validateEdge = (
    edge: Edge,
    index: number,
    componentIds: string[]
  ): ValidationError[] => {
    const errors: ValidationError[] = [];
    const prefix = `edges[${index}]`;

    // Source ID must exist
    if (!edge.sourceId) {
      errors.push({ field: `${prefix}.sourceId`, message: 'Source ID is required' });
    } else if (!componentIds.includes(edge.sourceId)) {
      errors.push({ field: `${prefix}.sourceId`, message: `Component ${edge.sourceId} not found` });
    }

    // Target ID must exist
    if (!edge.targetId) {
      errors.push({ field: `${prefix}.targetId`, message: 'Target ID is required' });
    } else if (!componentIds.includes(edge.targetId)) {
      errors.push({ field: `${prefix}.targetId`, message: `Component ${edge.targetId} not found` });
    }

    // No self-loops
    if (edge.sourceId === edge.targetId) {
      errors.push({ field: `${prefix}`, message: 'Self-loops are not allowed' });
    }

    // Edge type required
    if (!edge.edgeType) {
      errors.push({ field: `${prefix}.edgeType`, message: 'Edge type is required' });
    }

    // Diameter: 0-500
    if (edge.diameterMm !== undefined) {
      if (edge.diameterMm < 0) {
        errors.push({ field: `${prefix}.diameterMm`, message: 'Diameter must be ≥0' });
      } else if (edge.diameterMm > 500) {
        errors.push({ field: `${prefix}.diameterMm`, message: 'Diameter must be ≤500 mm' });
      }
    }

    // Length: 0-1000
    if (edge.lengthM !== undefined) {
      if (edge.lengthM < 0) {
        errors.push({ field: `${prefix}.lengthM`, message: 'Length must be ≥0' });
      } else if (edge.lengthM > 1000) {
        errors.push({ field: `${prefix}.lengthM`, message: 'Length must be ≤1000 m' });
      }
    }

    // Pressure rating: 0-1000
    if (edge.pressureRatingBar !== undefined) {
      if (edge.pressureRatingBar < 0) {
        errors.push({ field: `${prefix}.pressureRatingBar`, message: 'Pressure rating must be ≥0' });
      } else if (edge.pressureRatingBar > 1000) {
        errors.push({ field: `${prefix}.pressureRatingBar`, message: 'Pressure rating must be ≤1000 bar' });
      }
    }

    return errors;
  };

  /**
   * Validate edges array
   */
  const validateEdges = (edges: Edge[], components: Component[]): ValidationError[] => {
    let errors: ValidationError[] = [];

    // Min 1 edge required
    if (edges.length < 1) {
      errors.push({ field: 'edges', message: 'At least 1 edge is required' });
      return errors;
    }

    const componentIds = components.map(c => c.componentId);

    // Validate each edge
    edges.forEach((edge, index) => {
      errors = errors.concat(validateEdge(edge, index, componentIds));
    });

    return errors;
  };

  /**
   * Validate complete graph topology
   */
  const validateTopology = (topology: GraphTopology): ValidationError[] => {
    let errors: ValidationError[] = [];

    // Validate equipment
    errors = errors.concat(validateEquipment(topology));

    // Validate components
    if (topology.components && topology.components.length > 0) {
      errors = errors.concat(validateComponents(topology.components));
    } else {
      errors.push({ field: 'components', message: 'At least 2 components are required' });
    }

    // Validate edges
    if (topology.edges && topology.edges.length > 0) {
      errors = errors.concat(validateEdges(topology.edges, topology.components || []));
    } else {
      errors.push({ field: 'edges', message: 'At least 1 edge is required' });
    }

    return errors;
  };

  return {
    validateEquipment,
    validateComponent,
    validateComponents,
    validateEdge,
    validateEdges,
    validateTopology,
  };
};
