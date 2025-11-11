"""
Tests for equipment metadata API
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_equipment(client: AsyncClient, test_user):
    """Test creating new equipment"""
    equipment_data = {
        "system_id": "test_press_01",
        "system_type": "press",
        "name": "Test Hydraulic Press",
        "adjacency_matrix": [[0, 1], [1, 0]],
        "components": [
            {
                "component_id": "pump",
                "component_type": "pump",
                "name": "Main Pump",
                "sensors": ["pressure", "flow"],
                "normal_ranges": {
                    "pressure": {"min": 10, "max": 250},
                    "flow": {"min": 0, "max": 100}
                }
            },
            {
                "component_id": "valve",
                "component_type": "valve",
                "name": "Control Valve",
                "sensors": ["position", "pressure"],
                "normal_ranges": {
                    "position": {"min": 0, "max": 100}
                },
                "connected_to": ["pump"]
            }
        ]
    }

    response = await client.post(
        "/metadata/",
        json=equipment_data,
        headers={"X-API-Key": test_user.api_key}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["system_id"] == "test_press_01"
    assert data["system_type"] == "press"
    assert len(data["components"]) == 2


@pytest.mark.asyncio
async def test_get_equipment(client: AsyncClient, test_user):
    """Test retrieving equipment"""
    # First create equipment
    equipment_data = {...}  # Same as above
    await client.post("/metadata/", json=equipment_data, headers={"X-API-Key": test_user.api_key})

    # Then retrieve it
    response = await client.get(
        "/metadata/test_press_01",
        headers={"X-API-Key": test_user.api_key}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["system_id"] == "test_press_01"
