import pytest
from apps.diagnostics.models import HydraulicSystem, SystemComponent
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

User = get_user_model()


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(email="u@e.com", username="u", password="Pwd12345")


@pytest.fixture
def auth_client(api_client, user):
    resp = api_client.post(
        reverse("token_obtain_pair"), {"email": user.email, "password": "Pwd12345"}
    )
    token = resp.data["access"]
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    return api_client


@pytest.mark.django_db
def test_hydraulic_system_crud(auth_client):
    # Create
    data = {
        "name": "HS1",
        "system_type": "industrial",
        "status": "active",
        "criticality": "medium",
    }
    resp = auth_client.post("/api/diagnostics/systems/", data)
    assert resp.status_code == status.HTTP_201_CREATED
    hs_id = resp.data["id"]

    # Retrieve
    resp2 = auth_client.get(f"/api/diagnostics/systems/{hs_id}/")
    assert resp2.status_code == status.HTTP_200_OK
    assert resp2.data["name"] == "HS1"

    # Update
    upd = {"status": "maintenance"}
    resp3 = auth_client.patch(f"/api/diagnostics/systems/{hs_id}/", upd)
    assert resp3.status_code == status.HTTP_200_OK
    assert resp3.data["status"] == "maintenance"

    # Delete
    resp4 = auth_client.delete(f"/api/diagnostics/systems/{hs_id}/")
    assert resp4.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.django_db
def test_system_component_flow(auth_client):
    hs = HydraulicSystem.objects.create(
        name="HS2", system_type="mobile", status="active", criticality="low"
    )
    # Create component
    data = {"system": hs.id, "name": "Comp1", "specification": {"rpm": 1000}}
    r1 = auth_client.post("/api/diagnostics/components/", data, format="json")
    assert r1.status_code == status.HTTP_201_CREATED
    comp_id = r1.data["id"]
    # List and detail
    r2 = auth_client.get("/api/diagnostics/components/")
    assert any(c["id"] == comp_id for c in r2.data)
    r3 = auth_client.get(f"/api/diagnostics/components/{comp_id}/")
    assert r3.data["name"] == "Comp1"


@pytest.mark.django_db
def test_sensor_data_and_diagnostic(auth_client):
    hs = HydraulicSystem.objects.create(
        name="HS3", system_type="marine", status="active", criticality="high"
    )
    comp = SystemComponent.objects.create(system=hs, name="C1")
    # Post sensor data
    sdata = {"system": hs.id, "component": comp.id, "value": 12.5, "unit": "bar"}
    r1 = auth_client.post("/api/diagnostics/sensor-data/", sdata)
    assert r1.status_code == status.HTTP_405_METHOD_NOT_ALLOWED  # read-only

    # Use viewset action to upload via CSV simulation skipped

    # Diagnostic report creation via action
    r2 = auth_client.post(f"/api/diagnostics/systems/{hs.id}/diagnose/")
    assert r2.status_code == status.HTTP_201_CREATED
    assert r2.data["status"] == "pending"


@pytest.mark.django_db
def test_reports_and_actions(auth_client):
    hs = HydraulicSystem.objects.create(
        name="HS4", system_type="construction", status="active", criticality="medium"
    )
    # Create report
    r = auth_client.post(f"/api/diagnostics/systems/{hs.id}/diagnose/")
    rid = r.data["id"]
    # Complete
    rc = auth_client.post(f"/api/diagnostics/reports/{rid}/complete/")
    assert rc.status_code == status.HTTP_200_OK
    assert rc.data["status"] == "completed"


@pytest.mark.django_db
def test_maintenance_schedule(auth_client):
    hs = HydraulicSystem.objects.create(
        name="HS5", system_type="aviation", status="active", criticality="critical"
    )
    ms_data = {
        "system": hs.id,
        "schedule_date": "2025-12-01",
        "description": "Year-end",
    }
    r1 = auth_client.post("/api/diagnostics/maintenance/", ms_data)
    assert r1.status_code == status.HTTP_201_CREATED
    ms_id = r1.data["id"]
    # List and delete
    r2 = auth_client.get("/api/diagnostics/maintenance/")
    assert any(m["id"] == ms_id for m in r2.data)
    r3 = auth_client.delete(f"/api/diagnostics/maintenance/{ms_id}/")
    assert r3.status_code == status.HTTP_204_NO_CONTENT
