"""Data generation module for hydraulic scenario synthesis.

This module provides tools for generating realistic hydraulic system scenarios
for training Universal Temporal GNN models.

Components:
    - HydraulicScenarioGenerator: Main generator class
    - TopologyDefinitions: Hydraulic circuit topologies
    - FeatureDefinitions: Node and edge feature specifications
    - PhysicalValidators: Physics-based validation
"""

from .feature_definitions import EdgeFeatures, NodeFeatures
from .hydraulic_scenario_generator import HydraulicScenarioGenerator
from .topologies import TopologyDefinitions, TopologyType
from .validators import PhysicalValidator

__all__ = [
    'HydraulicScenarioGenerator',
    'TopologyDefinitions',
    'TopologyType',
    'NodeFeatures',
    'EdgeFeatures',
    'PhysicalValidator',
]
