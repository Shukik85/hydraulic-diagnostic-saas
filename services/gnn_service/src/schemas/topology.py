"""Topology management schemas for hydraulic systems.

Provides schemas for:
- Pre-defined topology templates
- User-defined topology configurations
- Edge configurations with static properties
- Built-in templates for common hydraulic systems

Usage:
    # Use pre-defined template
    template = TOPOLOGY_TEMPLATES["standard_pump_system"]
    config = TopologyConfig.from_template(template)
    
    # Create custom topology
    config = TopologyConfig(
        topology_id="my_system",
        components=[...],
        edges=[...]
    )

Author: GNN Service Team
Python: 3.14+
Pydantic: 2.x
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .graph import ComponentType, EdgeMaterial


class EdgeConfiguration(BaseModel):
    """Static edge configuration (connection properties).
    
    Represents physical properties of connections (pipes, hoses) that
    don't change during operation. Dynamic properties (flow, pressure drop)
    are computed at inference time.
    
    Attributes:
        source_id: Source component ID
        target_id: Target component ID
        diameter_mm: Pipe/hose diameter in millimeters
        length_m: Connection length in meters
        material: Pipe/hose material type
        pressure_rating_bar: Maximum pressure rating in bar
        install_date: Installation date (for age computation)
        last_maintenance_date: Last maintenance date (for maintenance score)
    
    Example:
        >>> edge = EdgeConfiguration(
        ...     source_id="pump_1",
        ...     target_id="valve_1",
        ...     diameter_mm=25.0,
        ...     length_m=5.2,
        ...     material=EdgeMaterial.STEEL,
        ...     pressure_rating_bar=250.0,
        ...     install_date=date(2020, 1, 15)
        ... )
    """
    
    source_id: str = Field(description="Source component ID")
    target_id: str = Field(description="Target component ID")
    diameter_mm: float = Field(gt=0, le=500, description="Diameter in mm")
    length_m: float = Field(gt=0, le=1000, description="Length in meters")
    material: EdgeMaterial = Field(description="Pipe/hose material")
    pressure_rating_bar: float = Field(gt=0, description="Max pressure in bar")
    install_date: Optional[date] = Field(default=None, description="Installation date")
    last_maintenance_date: Optional[date] = Field(default=None, description="Last maintenance")
    
    @field_validator('last_maintenance_date')
    @classmethod
    def validate_maintenance_after_install(cls, v: Optional[date], info) -> Optional[date]:
        """Ensure maintenance date is after installation."""
        if v and info.data.get('install_date'):
            if v < info.data['install_date']:
                raise ValueError(
                    f"Maintenance date {v} cannot be before install date "
                    f"{info.data['install_date']}"
                )
        return v
    
    def get_age_hours(self, current_date: date | datetime) -> float:
        """Compute connection age in hours.
        
        Args:
            current_date: Current date/datetime
        
        Returns:
            Age in hours (0 if install_date unknown)
        """
        if not self.install_date:
            return 0.0
        
        if isinstance(current_date, datetime):
            current_date = current_date.date()
        
        delta = current_date - self.install_date
        return delta.days * 24.0
    
    def get_maintenance_score(self, current_date: date | datetime) -> float:
        """Compute maintenance score [0, 1].
        
        Score decays from 1.0 (just maintained) to 0.0 (365 days ago).
        Returns 0.5 if no maintenance history.
        
        Args:
            current_date: Current date/datetime
        
        Returns:
            Maintenance score in [0, 1]
        """
        if not self.last_maintenance_date:
            return 0.5  # Unknown = neutral
        
        if isinstance(current_date, datetime):
            current_date = current_date.date()
        
        days_since = (current_date - self.last_maintenance_date).days
        
        # Decay over 365 days
        score = max(0.0, 1.0 - days_since / 365.0)
        return score
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_id": "pump_1",
                    "target_id": "valve_1",
                    "diameter_mm": 25.0,
                    "length_m": 5.2,
                    "material": "steel",
                    "pressure_rating_bar": 250.0,
                    "install_date": "2020-01-15",
                    "last_maintenance_date": "2024-06-01"
                }
            ]
        }
    }


class ComponentConfiguration(BaseModel):
    """Component configuration in topology.
    
    Attributes:
        component_id: Unique component identifier
        component_type: Type of component (pump, valve, etc.)
        description: Human-readable description
    """
    
    component_id: str = Field(description="Unique component ID")
    component_type: ComponentType = Field(description="Component type")
    description: Optional[str] = Field(default=None, description="Component description")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "component_id": "pump_1",
                    "component_type": "pump",
                    "description": "Main hydraulic pump"
                }
            ]
        }
    }


class TopologyConfig(BaseModel):
    """Complete topology configuration for hydraulic system.
    
    Defines the structure of a hydraulic system: components and their
    connections. Used for:
    - Pre-configured systems (via topology_id)
    - Custom user-defined systems
    - Template instantiation
    
    Attributes:
        topology_id: Unique identifier for this topology
        name: Human-readable name
        description: Detailed description
        components: List of components in the system
        edges: List of connections between components
        created_at: Creation timestamp
        updated_at: Last update timestamp
    
    Example:
        >>> config = TopologyConfig(
        ...     topology_id="my_pump_system",
        ...     name="Single Pump System",
        ...     components=[...],
        ...     edges=[...]
        ... )
    """
    
    topology_id: str = Field(description="Unique topology identifier")
    name: str = Field(description="Human-readable name")
    description: Optional[str] = Field(default=None, description="Detailed description")
    components: list[ComponentConfiguration] = Field(description="System components")
    edges: list[EdgeConfiguration] = Field(description="Component connections")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update")
    
    @field_validator('edges')
    @classmethod
    def validate_edges_reference_components(cls, v: list[EdgeConfiguration], info) -> list[EdgeConfiguration]:
        """Ensure all edges reference existing components."""
        components = info.data.get('components', [])
        component_ids = {c.component_id for c in components}
        
        for edge in v:
            if edge.source_id not in component_ids:
                raise ValueError(f"Edge source '{edge.source_id}' not in components")
            if edge.target_id not in component_ids:
                raise ValueError(f"Edge target '{edge.target_id}' not in components")
        
        return v
    
    @classmethod
    def from_template(cls, template: TopologyTemplate, topology_id: str) -> TopologyConfig:
        """Create config from template.
        
        Args:
            template: Template to instantiate
            topology_id: Unique ID for new topology
        
        Returns:
            New topology configuration
        """
        return cls(
            topology_id=topology_id,
            name=template.name,
            description=template.description,
            components=template.components,
            edges=template.edges
        )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "topology_id": "system_001",
                    "name": "Standard Pump System",
                    "description": "Single pump with valve and filter",
                    "components": [
                        {"component_id": "pump_1", "component_type": "pump"},
                        {"component_id": "valve_1", "component_type": "valve"},
                        {"component_id": "filter_1", "component_type": "filter"}
                    ],
                    "edges": [
                        {
                            "source_id": "pump_1",
                            "target_id": "valve_1",
                            "diameter_mm": 25.0,
                            "length_m": 5.0,
                            "material": "steel",
                            "pressure_rating_bar": 250.0
                        }
                    ]
                }
            ]
        }
    }


class TopologyTemplate(BaseModel):
    """Pre-defined topology template.
    
    Templates provide common hydraulic system configurations that users
    can instantiate with custom topology_id.
    
    Attributes:
        template_id: Template identifier (used in API)
        name: Template name
        description: What this template represents
        components: Standard components in this template
        edges: Standard connections
        category: Template category (e.g., 'pump_systems')
    """
    
    template_id: str = Field(description="Template identifier")
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    components: list[ComponentConfiguration] = Field(description="Template components")
    edges: list[EdgeConfiguration] = Field(description="Template edges")
    category: str = Field(default="general", description="Template category")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "template_id": "standard_pump_system",
                    "name": "Standard Pump System",
                    "description": "Single pump with valve and filter",
                    "category": "pump_systems",
                    "components": [...],
                    "edges": [...]
                }
            ]
        }
    }


# ============================================================================
# Built-in Topology Templates
# ============================================================================

TOPOLOGY_TEMPLATES: dict[str, TopologyTemplate] = {
    "standard_pump_system": TopologyTemplate(
        template_id="standard_pump_system",
        name="Standard Pump System",
        description="Single main pump with valve and filter in series",
        category="pump_systems",
        components=[
            ComponentConfiguration(
                component_id="pump_1",
                component_type=ComponentType.PUMP,
                description="Main hydraulic pump"
            ),
            ComponentConfiguration(
                component_id="valve_1",
                component_type=ComponentType.VALVE,
                description="Control valve"
            ),
            ComponentConfiguration(
                component_id="filter_1",
                component_type=ComponentType.FILTER,
                description="Hydraulic filter"
            ),
        ],
        edges=[
            EdgeConfiguration(
                source_id="pump_1",
                target_id="valve_1",
                diameter_mm=25.0,
                length_m=5.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=250.0
            ),
            EdgeConfiguration(
                source_id="valve_1",
                target_id="filter_1",
                diameter_mm=25.0,
                length_m=3.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=250.0
            ),
        ]
    ),
    
    "dual_pump_system": TopologyTemplate(
        template_id="dual_pump_system",
        name="Dual Pump System with Redundancy",
        description="Two pumps with crossover valve for redundancy",
        category="pump_systems",
        components=[
            ComponentConfiguration(
                component_id="pump_1",
                component_type=ComponentType.PUMP,
                description="Primary pump"
            ),
            ComponentConfiguration(
                component_id="pump_2",
                component_type=ComponentType.PUMP,
                description="Backup pump"
            ),
            ComponentConfiguration(
                component_id="crossover_valve",
                component_type=ComponentType.VALVE,
                description="Crossover valve for redundancy"
            ),
            ComponentConfiguration(
                component_id="filter_1",
                component_type=ComponentType.FILTER,
                description="Main filter"
            ),
        ],
        edges=[
            EdgeConfiguration(
                source_id="pump_1",
                target_id="crossover_valve",
                diameter_mm=32.0,
                length_m=4.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=300.0
            ),
            EdgeConfiguration(
                source_id="pump_2",
                target_id="crossover_valve",
                diameter_mm=32.0,
                length_m=4.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=300.0
            ),
            EdgeConfiguration(
                source_id="crossover_valve",
                target_id="filter_1",
                diameter_mm=32.0,
                length_m=2.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=300.0
            ),
        ]
    ),
    
    "hydraulic_circuit_type_a": TopologyTemplate(
        template_id="hydraulic_circuit_type_a",
        name="Industrial Hydraulic Circuit Type A",
        description="Complete hydraulic circuit with pump, valve, cylinder, and filter",
        category="industrial_circuits",
        components=[
            ComponentConfiguration(
                component_id="pump_1",
                component_type=ComponentType.PUMP,
                description="Hydraulic pump"
            ),
            ComponentConfiguration(
                component_id="directional_valve_1",
                component_type=ComponentType.VALVE,
                description="4/3 directional control valve"
            ),
            ComponentConfiguration(
                component_id="cylinder_1",
                component_type=ComponentType.CYLINDER,
                description="Hydraulic cylinder"
            ),
            ComponentConfiguration(
                component_id="filter_1",
                component_type=ComponentType.FILTER,
                description="Return line filter"
            ),
        ],
        edges=[
            EdgeConfiguration(
                source_id="pump_1",
                target_id="directional_valve_1",
                diameter_mm=20.0,
                length_m=3.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=250.0
            ),
            EdgeConfiguration(
                source_id="directional_valve_1",
                target_id="cylinder_1",
                diameter_mm=16.0,
                length_m=8.0,
                material=EdgeMaterial.RUBBER,  # Flexible hose to cylinder
                pressure_rating_bar=200.0
            ),
            EdgeConfiguration(
                source_id="cylinder_1",
                target_id="filter_1",
                diameter_mm=20.0,
                length_m=6.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=100.0  # Return line, lower pressure
            ),
        ]
    ),
}


def get_template(template_id: str) -> TopologyTemplate:
    """Get built-in topology template.
    
    Args:
        template_id: Template identifier
    
    Returns:
        Topology template
    
    Raises:
        KeyError: If template not found
    
    Example:
        >>> template = get_template("standard_pump_system")
        >>> config = TopologyConfig.from_template(template, "my_system_01")
    """
    if template_id not in TOPOLOGY_TEMPLATES:
        available = ", ".join(TOPOLOGY_TEMPLATES.keys())
        raise KeyError(
            f"Template '{template_id}' not found. "
            f"Available templates: {available}"
        )
    return TOPOLOGY_TEMPLATES[template_id]


def list_templates(category: Optional[str] = None) -> list[TopologyTemplate]:
    """List available topology templates.
    
    Args:
        category: Optional category filter
    
    Returns:
        List of topology templates
    
    Example:
        >>> templates = list_templates(category="pump_systems")
        >>> for t in templates:
        ...     print(f"{t.template_id}: {t.name}")
    """
    templates = list(TOPOLOGY_TEMPLATES.values())
    
    if category:
        templates = [t for t in templates if t.category == category]
    
    return templates
