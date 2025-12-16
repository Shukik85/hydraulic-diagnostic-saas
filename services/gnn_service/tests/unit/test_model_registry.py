import pytest

from src.inference.model_registry import ModelConfig, ModelRegistry


def test_model_config_validation(tmp_path) -> None:
    model_path = tmp_path / "v1.ckpt"
    model_path.write_bytes(b"dummy")

    config = ModelConfig(path=str(model_path), version="v1", traffic=0.8)
    assert config.traffic == 0.8

    with pytest.raises(ValueError):
        ModelConfig(path=str(model_path), version="v1", traffic=1.5)

    with pytest.raises(FileNotFoundError):
        ModelConfig(path="nonexistent.ckpt", version="v1")


def test_model_registry_registration(tmp_path) -> None:
    registry = ModelRegistry()
    model_path = tmp_path / "v1.ckpt"
    model_path.write_bytes(b"dummy")
    config = ModelConfig(path=str(model_path), version="v1", traffic=1.0)
    registry.register("v1", config)
    assert "v1" in registry._models


def test_model_registry_traffic_routing(tmp_path) -> None:
    registry = ModelRegistry()
    p1 = tmp_path / "v1.ckpt"
    p2 = tmp_path / "v2.ckpt"
    p1.write_bytes(b"1")
    p2.write_bytes(b"2")

    registry.register("v1", ModelConfig(path=str(p1), version="v1", traffic=0.8))
    registry.register("v2", ModelConfig(path=str(p2), version="v2", traffic=0.2))

    version = registry.select_model("req_123")
    assert version in {"v1", "v2"}
    assert registry.select_model("req_123") == version


def test_model_registry_get_model_not_loaded() -> None:
    registry = ModelRegistry()
    with pytest.raises(KeyError):
        registry.get_model("v1")
