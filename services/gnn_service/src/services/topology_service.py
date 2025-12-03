"""Topology service for managing hydraulic system templates.

Provides access to built-in topology templates and validation
for custom user-defined topologies.

Features:
- Built-in templates (standard_pump_system, dual_pump_system, etc.)
- In-memory caching for performance
- Custom topology validation
- Thread-safe operations

Author: GNN Service Team
Python: 3.14+
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import TYPE_CHECKING

from src.schemas.topology import (
    BUILTIN_TEMPLATES,
    TopologyTemplate,
    get_builtin_template,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.schemas import GraphTopology

logger = logging.getLogger(__name__)


class TopologyService:
    """Service for managing hydraulic system topologies.

    Singleton service that provides:
    - Access to built-in topology templates
    - Custom topology validation
    - In-memory caching
    - Thread-safe operations

    Examples:
        >>> service = TopologyService.get_instance()
        >>>
        >>> # Get built-in template
        >>> template = service.get_template("standard_pump_system")
        >>> topology = template.to_graph_topology(equipment_id="pump_001")
        >>>
        >>> # List all templates
        >>> templates = service.list_templates()
        >>>
        >>> # Validate custom topology
        >>> is_valid = service.validate_topology(custom_topology)
    """

    _instance: TopologyService | None = None
    _lock: Lock = Lock()

    def __init__(self, templates_path: Path | None = None):
        """Initialize topology service.

        Args:
            templates_path: Optional path to templates JSON file
        """
        self.templates_path = templates_path
        self._cache: dict[str, TopologyTemplate] = {}
        self._cache_timestamps: dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=1)  # Cache for 1 hour
        self._custom_topologies: dict[str, GraphTopology] = {}

        # Load built-in templates
        self._load_builtin_templates()

        # Load custom templates from file (if provided)
        if templates_path and templates_path.exists():
            self._load_templates_from_file(templates_path)

        logger.info(f"TopologyService initialized with {len(self._cache)} templates")

    @classmethod
    def get_instance(
        cls, templates_path: Path | None = None, force_new: bool = False
    ) -> TopologyService:
        """Get singleton instance of TopologyService.

        Thread-safe singleton pattern.

        Args:
            templates_path: Optional path to templates JSON
            force_new: Force create new instance (for testing)

        Returns:
            TopologyService instance

        Examples:
            >>> service = TopologyService.get_instance()
            >>> # All subsequent calls return same instance
            >>> service2 = TopologyService.get_instance()
            >>> assert service is service2
        """
        if force_new or cls._instance is None:
            with cls._lock:
                if force_new or cls._instance is None:
                    cls._instance = cls(templates_path=templates_path)

        return cls._instance

    def _load_builtin_templates(self) -> None:
        """Load built-in topology templates.

        Loads 3 built-in templates:
        - standard_pump_system
        - dual_pump_system
        - hydraulic_circuit_type_a
        """
        for template_id in BUILTIN_TEMPLATES:
            try:
                template = get_builtin_template(template_id)
                self._cache[template_id] = template
                self._cache_timestamps[template_id] = datetime.now()
                logger.debug(f"Loaded built-in template: {template_id}")
            except ValueError as e:
                logger.warning(f"Failed to load template {template_id}: {e}")

    def _load_templates_from_file(self, path: Path) -> None:
        """Load custom templates from JSON file.

        Args:
            path: Path to JSON file with templates

        File format:
        {
            "templates": [
                {
                    "template_id": "custom_template_1",
                    "name": "Custom Template",
                    "description": "...",
                    "components": [...],
                    "edges": [...]
                }
            ]
        }
        """
        try:
            with open(path) as f:
                data = json.load(f)

            templates = data.get("templates", [])

            for template_data in templates:
                try:
                    template = TopologyTemplate(**template_data)
                    template_id = template.template_id

                    self._cache[template_id] = template
                    self._cache_timestamps[template_id] = datetime.now()

                    logger.info(f"Loaded custom template: {template_id}")
                except Exception as e:
                    logger.exception(f"Failed to parse template: {e}")

        except FileNotFoundError:
            logger.warning(f"Templates file not found: {path}")
        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON in templates file: {e}")

    def get_template(self, template_id: str) -> TopologyTemplate | None:
        """Get topology template by ID.

        Checks cache first, returns None if not found.

        Args:
            template_id: Template identifier

        Returns:
            TopologyTemplate if found, None otherwise

        Examples:
            >>> service = TopologyService.get_instance()
            >>> template = service.get_template("standard_pump_system")
            >>> if template:
            ...     topology = template.to_graph_topology("equipment_001")
        """
        # Check cache
        if template_id in self._cache:
            # Check if cache expired
            cached_time = self._cache_timestamps.get(template_id)
            if cached_time and (datetime.now() - cached_time) < self._cache_ttl:
                return self._cache[template_id]
            # Cache expired, remove
            del self._cache[template_id]
            del self._cache_timestamps[template_id]

        # Try to load built-in template
        if template_id in BUILTIN_TEMPLATES:
            try:
                template = get_builtin_template(template_id)
                self._cache[template_id] = template
                self._cache_timestamps[template_id] = datetime.now()
                return template
            except ValueError:
                pass

        logger.warning(f"Template not found: {template_id}")
        return None

    def list_templates(self) -> list[dict[str, str]]:
        """List all available templates.

        Returns:
            List of template metadata (id, name, description)

        Examples:
            >>> service = TopologyService.get_instance()
            >>> templates = service.list_templates()
            >>> for t in templates:
            ...     print(f"{t['template_id']}: {t['name']}")
        """
        templates = []

        for template in self._cache.values():
            templates.append(
                {
                    "template_id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "num_components": len(template.components),
                    "num_edges": len(template.edges),
                }
            )

        return templates

    def validate_topology(self, topology: GraphTopology) -> tuple[bool, list[str]]:
        """Validate custom topology.

        Checks:
        - All edges reference existing components
        - No duplicate component IDs
        - No self-loops (unless intentional)
        - Reasonable edge properties

        Args:
            topology: GraphTopology to validate

        Returns:
            Tuple of (is_valid, list_of_errors)

        Examples:
            >>> service = TopologyService.get_instance()
            >>> is_valid, errors = service.validate_topology(custom_topology)
            >>> if not is_valid:
            ...     print("Validation errors:", errors)
        """
        errors = []

        # Check components
        if not topology.components:
            errors.append("Topology must have at least one component")

        component_ids = set(topology.components.keys())

        # Check for duplicate component IDs (already handled by dict, but explicit)
        if len(component_ids) != len(topology.components):
            errors.append("Duplicate component IDs detected")

        # Check edges
        if not topology.edges:
            errors.append("Topology must have at least one edge")

        for i, edge in enumerate(topology.edges):
            # Check source exists
            if edge.source_id not in component_ids:
                errors.append(f"Edge {i}: source_id '{edge.source_id}' not in components")

            # Check target exists
            if edge.target_id not in component_ids:
                errors.append(f"Edge {i}: target_id '{edge.target_id}' not in components")

            # Check self-loop (warn, not error)
            if edge.source_id == edge.target_id:
                logger.warning(f"Edge {i} is a self-loop: {edge.source_id}")

            # Check edge properties
            if edge.diameter_mm <= 0:
                errors.append(f"Edge {i}: diameter_mm must be positive")

            if edge.length_m <= 0:
                errors.append(f"Edge {i}: length_m must be positive")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"Topology validation passed: {topology.equipment_id}")
        else:
            logger.warning(f"Topology validation failed: {topology.equipment_id}, errors: {errors}")

        return is_valid, errors

    def register_custom_topology(self, topology_id: str, topology: GraphTopology) -> bool:
        """Register custom topology for reuse.

        Validates topology before registering.

        Args:
            topology_id: Unique identifier for topology
            topology: GraphTopology to register

        Returns:
            True if registered successfully, False otherwise

        Examples:
            >>> service = TopologyService.get_instance()
            >>> success = service.register_custom_topology(
            ...     "my_custom_system",
            ...     custom_topology
            ... )
        """
        # Validate first
        is_valid, errors = self.validate_topology(topology)

        if not is_valid:
            logger.error(f"Cannot register invalid topology '{topology_id}': {errors}")
            return False

        # Register
        self._custom_topologies[topology_id] = topology
        logger.info(f"Registered custom topology: {topology_id}")

        return True

    def get_custom_topology(self, topology_id: str) -> GraphTopology | None:
        """Get registered custom topology.

        Args:
            topology_id: Custom topology identifier

        Returns:
            GraphTopology if found, None otherwise
        """
        return self._custom_topologies.get(topology_id)

    def clear_cache(self) -> None:
        """Clear topology cache.

        Useful for testing or forcing reload.
        """
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Topology cache cleared")

    def get_stats(self) -> dict[str, int]:
        """Get service statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cached_templates": len(self._cache),
            "custom_topologies": len(self._custom_topologies),
            "builtin_templates": len(BUILTIN_TEMPLATES),
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def get_topology_service(templates_path: Path | None = None) -> TopologyService:
    """Get TopologyService singleton instance.

    Convenience function for dependency injection.

    Args:
        templates_path: Optional path to custom templates

    Returns:
        TopologyService instance

    Examples:
        >>> service = get_topology_service()
        >>> template = service.get_template("standard_pump_system")
    """
    return TopologyService.get_instance(templates_path=templates_path)
