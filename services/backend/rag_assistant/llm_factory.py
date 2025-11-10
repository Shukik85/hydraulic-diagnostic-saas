"""Фабрика для создания LLM и моделей эмбеддингов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.conf import settings

# LangChain Ollama provider (modern imports)
from langchain_ollama import (
    ChatOllama,  # type: ignore
    OllamaEmbeddings,  # type: ignore
)


@dataclass
class LLMConfig:
    """Конфигурация для языковых моделей."""

    provider: str = getattr(settings, "LLM_PROVIDER", "ollama")
    model: str = getattr(settings, "LLM_MODEL", "qwen3:8b")
    temperature: float = float(getattr(settings, "LLM_TEMPERATURE", 0.1))


@dataclass
class EmbeddingConfig:
    """Конфигурация для моделей эмбеддингов."""

    provider: str = getattr(settings, "EMBEDDING_PROVIDER", "ollama")
    model: str = getattr(settings, "EMBEDDING_MODEL", "nomic-embed-text")


class LLMFactory:
    """Фабрика для создания LLM и моделей эмбеддингов."""

    @staticmethod
    def create_chat_model(cfg: LLMConfig | None = None) -> Any:
        """Создает чат-модель на основе конфигурации.

        Args:
            cfg: Конфигурация LLM (опционально)

        Returns:
            Инициализированная чат-модель

        Raises:
            ValueError: Если провайдер не поддерживается
        """
        cfg = cfg or LLMConfig()
        if cfg.provider == "ollama":
            return ChatOllama(model=cfg.model, temperature=cfg.temperature)
        raise ValueError(f"Unsupported LLM provider: {cfg.provider}")

    @staticmethod
    def create_embedder(cfg: EmbeddingConfig | None = None) -> Any:
        """Создает модель эмбеддингов на основе конфигурации.

        Args:
            cfg: Конфигурация эмбеддингов (опционально)

        Returns:
            Инициализированная модель эмбеддингов

        Raises:
            ValueError: Если провайдер не поддерживается
        """
        cfg = cfg or EmbeddingConfig()
        if cfg.provider == "ollama":
            return OllamaEmbeddings(model=cfg.model)
        raise ValueError(f"Unsupported Embedding provider: {cfg.provider}")
