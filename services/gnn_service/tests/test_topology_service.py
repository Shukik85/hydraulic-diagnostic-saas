"""Tests for TopologyService.

Tests topology template management:
- Loading built-in templates
- Custom topology registration
- Validation
- Caching

Author: GNN Service Team
Python: 3.14+
"""


import pytest

from src.schemas import ComponentSpec, ComponentType, EdgeSpec, EdgeType, GraphTopology
from src.schemas.graph import EdgeMaterial
from src.services.topology_service import TopologyService, get_topology_service


class TestTopologyService:
    """Test TopologyService class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset singleton between tests."""
        TopologyService._instance = None
        yield
        TopologyService._instance = None

    def test_singleton_pattern(self):
        """Test TopologyService is a singleton."""
        service1 = TopologyService.get_instance()
        service2 = TopologyService.get_instance()

        assert service1 is service2

    def test_force_new_instance(self):
        """Test force_new parameter creates new instance."""
        service1 = TopologyService.get_instance()
        service2 = TopologyService.get_instance(force_new=True)

        assert service1 is not service2

    def test_load_builtin_templates(self):
        """Test built-in templates are loaded on init."""
        service = TopologyService.get_instance()

        templates = service.list_templates()

        # Should have 3 built-in templates
        assert len(templates) >= 3

        # Check template IDs
        template_ids = [t["template_id"] for t in templates]
        assert "standard_pump_system" in template_ids
        assert "dual_pump_system" in template_ids
        assert "hydraulic_circuit_type_a" in template_ids

    def test_get_template_by_id(self):
        """Test getting template by ID."""
        service = TopologyService.get_instance()

        template = service.get_template("standard_pump_system")

        assert template is not None
        assert template.template_id == "standard_pump_system"
        assert len(template.components) > 0
        assert len(template.edges) > 0

    def test_get_nonexistent_template(self):
        """Test getting non-existent template returns None."""
        service = TopologyService.get_instance()

        template = service.get_template("nonexistent_template")

        assert template is None

    def test_list_templates(self):
        """Test listing all templates."""
        service = TopologyService.get_instance()

        templates = service.list_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0

        # Check structure
        for template in templates:
            assert "template_id" in template
            assert "name" in template
            assert "description" in template
            assert "num_components" in template
            assert "num_edges" in template


class TestTopologyValidation:
    """Test topology validation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset singleton between tests."""
        TopologyService._instance = None
        yield
        TopologyService._instance = None

    @pytest.fixture
    def valid_topology(self):
        """Create valid topology."""
        components = {
            "pump_1": ComponentSpec(
                component_id="pump_1",
                component_type=ComponentType.PUMP,
                manufacturer="TestCo",
                model="P1"
            ),
            "valve_1": ComponentSpec(
                component_id="valve_1",
                component_type=ComponentType.VALVE,
                manufacturer="TestCo",
                model="V1"
            )
        }

        edges = [
            EdgeSpec(
                source_id="pump_1",
                target_id="valve_1",
                edge_type=EdgeType.PIPE,
                diameter_mm=25.0,
                length_m=2.0,
                material=EdgeMaterial.STEEL
            )
        ]

        return GraphTopology(
            equipment_id="test_equipment",
            components=components,
            edges=edges
        )

    def test_validate_valid_topology(self, valid_topology):
        """Test validating valid topology."""
        service = TopologyService.get_instance()

        is_valid, errors = service.validate_topology(valid_topology)

        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_component(self):
        """Test validation fails for missing component reference."""
        components = {
            "pump_1": ComponentSpec(
                component_id="pump_1",
                component_type=ComponentType.PUMP,
                manufacturer="TestCo",
                model="P1"
            )
        }

        edges = [
            EdgeSpec(
                source_id="pump_1",
                target_id="nonexistent_valve",  # Doesn't exist
                edge_type=EdgeType.PIPE,
                diameter_mm=25.0,
                length_m=2.0,
                material=EdgeMaterial.STEEL
            )
        ]

        topology = GraphTopology(
            equipment_id="test_equipment",
            components=components,
            edges=edges
        )

        service = TopologyService.get_instance()
        is_valid, errors = service.validate_topology(topology)

        assert not is_valid
        assert len(errors) > 0
        assert any("nonexistent_valve" in str(e) for e in errors)

    def test_validate_empty_components(self):
        """Test validation fails for empty components."""
        topology = GraphTopology(
            equipment_id="test_equipment",
            components={},
            edges=[]
        )

        service = TopologyService.get_instance()
        is_valid, errors = service.validate_topology(topology)

        assert not is_valid
        assert any("at least one component" in str(e) for e in errors)

    def test_validate_invalid_edge_properties(self):
        """Test validation fails for invalid edge properties."""
        components = {
            "pump_1": ComponentSpec(
                component_id="pump_1",
                component_type=ComponentType.PUMP,
                manufacturer="TestCo",
                model="P1"
            ),
            "valve_1": ComponentSpec(
                component_id="valve_1",
                component_type=ComponentType.VALVE,
                manufacturer="TestCo",
                model="V1"
            )
        }

        edges = [
            EdgeSpec(
                source_id="pump_1",
                target_id="valve_1",
                edge_type=EdgeType.PIPE,
                diameter_mm=-10.0,  # Invalid: negative
                length_m=2.0,
                material=EdgeMaterial.STEEL
            )
        ]

        topology = GraphTopology(
            equipment_id="test_equipment",
            components=components,
            edges=edges
        )

        service = TopologyService.get_instance()
        is_valid, errors = service.validate_topology(topology)

        assert not is_valid
        assert any("diameter" in str(e) for e in errors)


class TestCustomTopologies:
    """Test custom topology management."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset singleton between tests."""
        TopologyService._instance = None
        yield
        TopologyService._instance = None

    @pytest.fixture
    def custom_topology(self):
        """Create custom topology."""
        components = {
            "pump_custom": ComponentSpec(
                component_id="pump_custom",
                component_type=ComponentType.PUMP,
                manufacturer="CustomCo",
                model="CP1"
            ),
            "valve_custom": ComponentSpec(
                component_id="valve_custom",
                component_type=ComponentType.VALVE,
                manufacturer="CustomCo",
                model="CV1"
            )
        }

        edges = [
            EdgeSpec(
                source_id="pump_custom",
                target_id="valve_custom",
                edge_type=EdgeType.PIPE,
                diameter_mm=20.0,
                length_m=1.5,
                material=EdgeMaterial.STEEL
            )
        ]

        return GraphTopology(
            equipment_id="custom_equipment",
            components=components,
            edges=edges
        )

    def test_register_custom_topology(self, custom_topology):
        """Test registering custom topology."""
        service = TopologyService.get_instance()

        success = service.register_custom_topology(
            "my_custom_system",
            custom_topology
        )

        assert success

    def test_get_custom_topology(self, custom_topology):
        """Test retrieving custom topology."""
        service = TopologyService.get_instance()

        service.register_custom_topology("my_custom_system", custom_topology)

        retrieved = service.get_custom_topology("my_custom_system")

        assert retrieved is not None
        assert retrieved.equipment_id == custom_topology.equipment_id

    def test_register_invalid_topology(self):
        """Test registering invalid topology fails."""
        # Create invalid topology
        topology = GraphTopology(
            equipment_id="invalid",
            components={},  # Empty
            edges=[]
        )

        service = TopologyService.get_instance()
        success = service.register_custom_topology("invalid_system", topology)

        assert not success


class TestCaching:
    """Test caching behavior."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset singleton between tests."""
        TopologyService._instance = None
        yield
        TopologyService._instance = None

    def test_cache_hit(self):
        """Test cache is used for repeated requests."""
        service = TopologyService.get_instance()

        # First access
        template1 = service.get_template("standard_pump_system")

        # Second access (should hit cache)
        template2 = service.get_template("standard_pump_system")

        # Should be same object from cache
        assert template1 is template2

    def test_clear_cache(self):
        """Test clearing cache."""
        service = TopologyService.get_instance()

        # Load template
        service.get_template("standard_pump_system")

        # Clear cache
        service.clear_cache()

        # Cache should be empty
        stats = service.get_stats()
        assert stats["cached_templates"] == 0

    def test_get_stats(self):
        """Test getting service statistics."""
        service = TopologyService.get_instance()

        stats = service.get_stats()

        assert "cached_templates" in stats
        assert "custom_topologies" in stats
        assert "builtin_templates" in stats
        assert stats["builtin_templates"] >= 3


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset singleton between tests."""
        TopologyService._instance = None
        yield
        TopologyService._instance = None

    def test_get_topology_service(self):
        """Test convenience function."""
        service = get_topology_service()

        assert isinstance(service, TopologyService)

    def test_get_topology_service_singleton(self):
        """Test convenience function returns singleton."""
        service1 = get_topology_service()
        service2 = get_topology_service()

        assert service1 is service2
