import pytest
@pytest.fixture(autouse=True)
def set_root(monkeypatch):
    monkeypatch.chdir("services/gnn_service")
