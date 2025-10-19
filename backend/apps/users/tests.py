import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

User = get_user_model()


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def user_data():
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "StrongPass123",
        "first_name": "Test",
        "last_name": "User",
    }


@pytest.mark.django_db
def test_user_registration(api_client, user_data):
    url = reverse("user-register")
    response = api_client.post(url, data=user_data)
    assert response.status_code == status.HTTP_201_CREATED
    assert User.objects.filter(email=user_data["email"]).exists()


@pytest.mark.django_db
def test_login_and_token_refresh(api_client, user_data):
    # Register
    api_client.post(reverse("user-register"), data=user_data)
    # Login
    login_url = reverse("token_obtain_pair")
    resp = api_client.post(
        login_url, {"email": user_data["email"], "password": user_data["password"]}
    )
    assert resp.status_code == status.HTTP_200_OK
    assert "access" in resp.data and "refresh" in resp.data
    refresh = resp.data["refresh"]
    # Refresh
    refresh_url = reverse("token_refresh")
    resp2 = api_client.post(refresh_url, {"refresh": refresh})
    assert resp2.status_code == status.HTTP_200_OK
    assert "access" in resp2.data


@pytest.mark.django_db
def test_change_password(api_client, user_data):
    # Register and login
    api_client.post(reverse("user-register"), data=user_data)
    login_resp = api_client.post(
        reverse("token_obtain_pair"),
        {"email": user_data["email"], "password": user_data["password"]},
    )
    token = login_resp.data["access"]
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    # Change password
    url = reverse("users-change-password", args=[login_resp.data["user"]["id"]])
    new_pass = "NewStrong123"
    resp = api_client.post(
        url, {"old_password": user_data["password"], "new_password": new_pass}
    )
    assert resp.status_code == status.HTTP_200_OK
    # Re-login with new password
    api_client.credentials()  # clear
    resp2 = api_client.post(
        reverse("token_obtain_pair"),
        {"email": user_data["email"], "password": new_pass},
    )
    assert resp2.status_code == status.HTTP_200_OK


@pytest.mark.django_db
def test_profile_crud(api_client, user_data):
    # Register & login
    api_client.post(reverse("user-register"), data=user_data)
    login_resp = api_client.post(
        reverse("token_obtain_pair"),
        {"email": user_data["email"], "password": user_data["password"]},
    )
    token = login_resp.data["access"]
    client = api_client
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    # Retrieve profile (should exist)
    resp = client.get(reverse("profile-list"))
    assert resp.status_code == status.HTTP_200_OK
    assert isinstance(resp.data, list)
    # Update profile
    profile_id = resp.data[0]["id"]
    update_data = {"bio": "Updated bio", "location": "TestCity"}
    resp2 = client.patch(reverse("profile-detail", args=[profile_id]), data=update_data)
    assert resp2.status_code == status.HTTP_200_OK
    assert resp2.data["bio"] == "Updated bio"
    # Delete profile
    resp3 = client.delete(reverse("profile-detail", args=[profile_id]))
    assert resp3.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.django_db
def test_user_activity_logs(api_client, user_data):
    # Register & login
    api_client.post(reverse("user-register"), data=user_data)
    login_resp = api_client.post(
        reverse("token_obtain_pair"),
        {"email": user_data["email"], "password": user_data["password"]},
    )
    token = login_resp.data["access"]
    client = api_client
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    # Log in should create activity
    resp = client.get(reverse("activity-list"))
    assert resp.status_code == status.HTTP_200_OK
    assert all(log["user"]["email"] == user_data["email"] for log in resp.data)
