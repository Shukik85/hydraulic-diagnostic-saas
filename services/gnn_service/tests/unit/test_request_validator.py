"""Unit tests for RequestValidator."""

import pytest
from unittest.mock import Mock
from fastapi import HTTPException

from src.api.validators import ModelRequirements, RequestValidator
from src.schemas import GraphTopology
from src.schemas.requests import MinimalInferenceRequest


@pytest.fixture
def requirements() -> ModelRequirements:
    return ModelRequirements(
        node_feature_dim=34,
        edge_feature_dim=14,
        min_nodes=2,
        max_nodes=100,
    )


@pytest.fixture
def mock_topology_service():
    service = Mock()
    template = Mock()
    topology = Mock(spec=GraphTopology)
    topology.topology_id = "pump_v1"
    topology.num_components = 5
    topology.components = {
        "pump_1": Mock(component_id="pump_1"),
        "valve_1": Mock(component_id="valve_1"),
    }
    template.to_graph_topology = Mock(return_value=topology)
    service.get_template = Mock(return_value=template)
    service.get_all_templates = Mock(return_value={"pump_v1": template})
    return service


@pytest.fixture
def validator(requirements, mock_topology_service) -> RequestValidator:
    return RequestValidator(
        requirements=requirements,
        topology_service=mock_topology_service,
        max_batch_size=32,
    )


@pytest.fixture
def valid_request() -> MinimalInferenceRequest:
    return MinimalInferenceRequest(
        equipment_id="pump_001",
        topology_id="pump_v1",
        timestamp="2025-01-01T00:00:00Z",
        sensor_readings={
            "pump_1": {},
            "valve_1": {},
        },
    )


def test_validate_batch_size_valid(validator: RequestValidator) -> None:
    validator._validate_batch_size(10)
    validator._validate_batch_size(1)
    validator._validate_batch_size(32)


def test_validate_batch_size_too_large(validator: RequestValidator) -> None:
    with pytest.raises(HTTPException) as exc_info:
        validator._validate_batch_size(100)
    assert exc_info.value.status_code == 413


def test_validate_batch_size_too_small(validator: RequestValidator) -> None:
    with pytest.raises(HTTPException) as exc_info:
        validator._validate_batch_size(0)
    assert exc_info.value.status_code == 400


def test_validate_graph_size_valid(validator: RequestValidator) -> None:
    topology = Mock()
    topology.num_components = 50
    validator._validate_graph_size(topology)


def test_validate_graph_size_too_large(validator: RequestValidator) -> None:
    topology = Mock()
    topology.num_components = 200
    with pytest.raises(HTTPException) as exc_info:
        validator._validate_graph_size(topology)
    assert exc_info.value.status_code == 413


def test_validate_graph_size_too_small(validator: RequestValidator) -> None:
    topology = Mock()
    topology.num_components = 1
    with pytest.raises(HTTPException) as exc_info:
        validator._validate_graph_size(topology)
    assert exc_info.value.status_code == 400


def test_validate_sensors_missing(validator: RequestValidator) -> None:
    request = MinimalInferenceRequest(
        equipment_id="pump_001",
        topology_id="pump_v1",
        timestamp="2025-01-01T00:00:00Z",
        sensor_readings={"pump_1": {}},  # missing valve_1
    )
    topology = Mock()
    topology.topology_id = "pump_v1"
    topology.components = {
        "pump_1": Mock(component_id="pump_1"),
        "valve_1": Mock(component_id="valve_1"),
    }

    with pytest.raises(HTTPException) as exc_info:
        validator._validate_sensors(request, topology)
    assert exc_info.value.status_code == 400
    assert "valve_1" in str(exc_info.value.detail)


def test_get_topology_not_found(validator: RequestValidator) -> None:
    validator.topology_service.get_template = Mock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        validator._get_topology("nonexistent", "eq1")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_validate_diagnosis_request_success(
    validator: RequestValidator, valid_request: MinimalInferenceRequest
) -> None:
    topology = await validator.validate_diagnosis_request(valid_request)
    assert topology.topology_id == "pump_v1"
