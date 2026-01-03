"""Physical validators for hydraulic system data.

Ensures generated data adheres to physical laws and engineering constraints.

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import logging

import torch

from .feature_definitions import EdgeFeatures, NodeFeatures

logger = logging.getLogger(__name__)


class PhysicalValidator:
    """Validates hydraulic system data against physical constraints."""
    
    def __init__(self, strict: bool = True) -> None:
        """Initialize validator.
        
        Args:
            strict: If True, raise exceptions on validation errors.
                   If False, only log warnings.
        """
        self.strict = strict
        self.node_ranges = NodeFeatures.get_physical_ranges()
        self.edge_ranges = EdgeFeatures.get_physical_ranges()
    
    def validate_node_features(
        self,
        node_features: torch.Tensor,
        feature_names: list[str] | None = None
    ) -> tuple[bool, list[str]]:
        """Validate node feature tensor.
        
        Args:
            node_features: Tensor of shape [num_nodes, num_features]
            feature_names: Optional list of feature names for detailed reporting
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if feature_names is None:
            feature_names = NodeFeatures.get_feature_names()
        
        errors = []
        
        # Check dimensions
        if node_features.shape[1] != NodeFeatures.dimension():
            errors.append(
                f"Expected {NodeFeatures.dimension()} features, "
                f"got {node_features.shape[1]}"
            )
            return False, errors
        
        # Check each feature range
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.node_ranges:
                continue
            
            min_val, max_val = self.node_ranges[feature_name]
            feature_data = node_features[:, i]
            
            # Check for NaN/Inf
            if torch.isnan(feature_data).any():
                errors.append(f"NaN values in feature '{feature_name}'")
            if torch.isinf(feature_data).any():
                errors.append(f"Inf values in feature '{feature_name}'")
            
            # Check range
            out_of_range = (
                (feature_data < min_val) | (feature_data > max_val)
            )
            if out_of_range.any():
                num_violations = out_of_range.sum().item()
                actual_min = feature_data.min().item()
                actual_max = feature_data.max().item()
                errors.append(
                    f"Feature '{feature_name}': {num_violations} values "
                    f"out of range [{min_val}, {max_val}]. "
                    f"Actual range: [{actual_min:.2f}, {actual_max:.2f}]"
                )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            if self.strict:
                raise ValueError(
                    "Node feature validation failed:\n" +
                    "\n".join(f"  - {e}" for e in errors)
                )
            else:
                for error in errors:
                    logger.warning("Validation warning: %s", error)
        
        return is_valid, errors
    
    def validate_edge_features(
        self,
        edge_features: torch.Tensor,
        feature_names: list[str] | None = None
    ) -> tuple[bool, list[str]]:
        """Validate edge feature tensor.
        
        Args:
            edge_features: Tensor of shape [num_edges, num_features]
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if feature_names is None:
            feature_names = EdgeFeatures.get_feature_names()
        
        errors = []
        
        # Check dimensions
        if edge_features.shape[1] != EdgeFeatures.dimension():
            errors.append(
                f"Expected {EdgeFeatures.dimension()} features, "
                f"got {edge_features.shape[1]}"
            )
            return False, errors
        
        # Check each feature range
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.edge_ranges:
                continue
            
            min_val, max_val = self.edge_ranges[feature_name]
            feature_data = edge_features[:, i]
            
            # Check for NaN/Inf
            if torch.isnan(feature_data).any():
                errors.append(f"NaN values in feature '{feature_name}'")
            if torch.isinf(feature_data).any():
                errors.append(f"Inf values in feature '{feature_name}'")
            
            # Check range
            out_of_range = (
                (feature_data < min_val) | (feature_data > max_val)
            )
            if out_of_range.any():
                num_violations = out_of_range.sum().item()
                actual_min = feature_data.min().item()
                actual_max = feature_data.max().item()
                errors.append(
                    f"Feature '{feature_name}': {num_violations} values "
                    f"out of range [{min_val}, {max_val}]. "
                    f"Actual range: [{actual_min:.2f}, {actual_max:.2f}]"
                )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            if self.strict:
                raise ValueError(
                    "Edge feature validation failed:\n" +
                    "\n".join(f"  - {e}" for e in errors)
                )
            else:
                for error in errors:
                    logger.warning("Validation warning: %s", error)
        
        return is_valid, errors
    
    def validate_hydraulic_laws(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> tuple[bool, list[str]]:
        """Validate hydraulic physical laws.
        
        Checks:
        1. Pressure continuity (pressure drops make sense)
        2. Flow conservation at nodes
        3. Power = Pressure × Flow consistency
        
        Args:
            node_features: Node feature tensor [num_nodes, 34]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge feature tensor [num_edges, 14]
            
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        
        # Extract relevant features
        node_pressure = node_features[:, 0]  # pressure
        node_flow = node_features[:, 1]  # flow_rate
        node_power = node_features[:, 9]  # power
        
        edge_flow = edge_features[:, 0]  # flow_rate
        edge_pressure_drop = edge_features[:, 1]  # pressure_drop
        
        # Check 1: Pressure continuity
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        pressure_diff = node_pressure[source_nodes] - node_pressure[target_nodes]
        expected_drop = edge_pressure_drop
        
        # Allow 10% tolerance
        pressure_error = torch.abs(pressure_diff - expected_drop) / (expected_drop + 1e-6)
        high_error = pressure_error > 0.1
        
        if high_error.any():
            warnings.append(
                f"Pressure continuity: {high_error.sum().item()} edges "
                f"have >10% error in pressure drop"
            )
        
        # Check 2: Flow conservation (simplified - should sum to ~0 at each node)
        # This is complex for real systems, just check magnitudes are reasonable
        flow_ratio = edge_flow / (node_flow[source_nodes] + 1e-6)
        unreasonable_flow = (flow_ratio > 1.5) | (flow_ratio < 0.5)
        
        if unreasonable_flow.any():
            warnings.append(
                f"Flow conservation: {unreasonable_flow.sum().item()} edges "
                f"have unreasonable flow ratios"
            )
        
        # Check 3: Power consistency (Power ≈ Pressure × Flow / 600)
        # Factor 600 converts bar·l/min to kW
        expected_power = node_pressure * node_flow / 600.0
        power_error = torch.abs(node_power - expected_power) / (expected_power + 1e-6)
        
        # Only check for pumps and motors (where power is relevant)
        # Indices 24-28 are component type one-hot encodings
        is_pump = node_features[:, 24] > 0.5
        is_motor = node_features[:, 26] > 0.5
        power_relevant = is_pump | is_motor
        
        if power_relevant.any():
            high_power_error = (power_error > 0.2) & power_relevant
            if high_power_error.any():
                warnings.append(
                    f"Power consistency: {high_power_error.sum().item()} "
                    f"pumps/motors have >20% power calculation error"
                )
        
        # Log warnings
        for warning in warnings:
            logger.warning("Physical law validation: %s", warning)
        
        return len(warnings) == 0, warnings
    
    def validate_graph(self, data) -> bool:
        """Validate complete PyG Data object.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            True if all validations pass
        """
        all_valid = True
        
        # Validate node features
        node_valid, _node_errors = self.validate_node_features(data.x)
        all_valid &= node_valid
        
        # Validate edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_valid, _edge_errors = self.validate_edge_features(data.edge_attr)
            all_valid &= edge_valid
            
            # Validate physical laws
            _laws_valid, _warnings = self.validate_hydraulic_laws(
                data.x, data.edge_index, data.edge_attr
            )
            # Physical laws generate warnings, not hard failures
        
        return all_valid
