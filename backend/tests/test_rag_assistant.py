import pytest
from apps.rag_assistant.models import Document, RagQueryLog, RagSystem
from apps.rag_assistant.rag_service import RagAssistant
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
    user = User.objects.create_user(email="u@e.com", username="u", password="Pwd12345")
    return user


@pytest.fixture
def auth_client(api_client, user):
    resp = api_client.post(
        reverse("token_obtain_pair"), {"email": user.email, "password": "Pwd12345"}
    )
    token = resp.data["access"]
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    return api_client


@pytest.fixture
def rag_system(db):
    return RagSystem.objects.create(
        name="test-system",
        description="Test",
        model_name="openai/gpt-3.5-turbo",
        index_type="faiss",
        index_config={},
    )


@pytest.mark.django_db
def test_document_crud(auth_client):
    # Create multilingual Markdown document
    data = {
        "title": "Привет",
        "content": "# Заголовок\nТекст на русском",
        "format": "md",
        "language": "ru",
        "metadata": {},
    }
    r1 = auth_client.post("/api/rag_assistant/documents/", data, format="json")
    assert r1.status_code == status.HTTP_201_CREATED
    doc_id = r1.data["id"]

    # Retrieve
    r2 = auth_client.get(f"/api/rag_assistant/documents/{doc_id}/")
    assert r2.status_code == status.HTTP_200_OK
    assert r2.data["language"] == "ru"

    # Update
    upd = {"language": "en", "content": "Hello"}
    r3 = auth_client.patch(f"/api/rag_assistant/documents/{doc_id}/", upd, format="json")
    assert r3.status_code == status.HTTP_200_OK
    assert r3.data["language"] == "en"

    # Delete
    r4 = auth_client.delete(f"/api/rag_assistant/documents/{doc_id}/")
    assert r4.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.django_db
def test_index_and_query(auth_client, rag_system, tmp_path):
    # Create document and associate
    _ = Document.objects.create(
        title="Hello",
        content="Hello world",
        format="txt",
        language="en",
        metadata={"rag_system": rag_system.id},
    )

    # Index via API
    r1 = auth_client.post(f"/api/rag_assistant/systems/{rag_system.id}/index/")
    # В текущей реализации index возвращает 202 (асинхронная задача)
    assert r1.status_code in (status.HTTP_200_OK, status.HTTP_202_ACCEPTED)

    # Query via API
    query_data = {"query": "world"}
    r2 = auth_client.post(
        f"/api/rag_assistant/systems/{rag_system.id}/query/", query_data, format="json"
    )
    assert r2.status_code == status.HTTP_200_OK
    assert "answer" in r2.data

    # Verify log created
    logs = RagQueryLog.objects.filter(system=rag_system)
    assert logs.exists()
    log = logs.first()
    assert log is not None
    assert log.query_text == "world"
    assert log.response_text is not None


@pytest.mark.django_db
def test_rag_assistant_service(tmp_path, rag_system):
    # Write temp file for loader
    doc = Document.objects.create(
        title="T", content="Sample text", format="txt", language="en", metadata={}
    )
    # Initialize assistant
    assistant = RagAssistant(rag_system)
    # Index document
    assistant.index_document(doc)
    # Ask question
    ans = assistant.answer("Sample")
    assert isinstance(ans, str)
