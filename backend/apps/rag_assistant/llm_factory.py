"""Модуль проекта с автогенерированным докстрингом."""

from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings

# LangChain Ollama provider (modern imports)
from langchain_ollama import (
    ChatOllama,  # type: ignore
    OllamaEmbeddings,  # type: ignore
)


@dataclass
class LLMConfig:
    provider: str = getattr(settings, "LLM_PROVIDER", "ollama")
    model: str = getattr(settings, "LLM_MODEL", "qwen3:8b")
    temperature: float = float(getattr(settings, "LLM_TEMPERATURE", 0.1))


@dataclass
class EmbeddingConfig:
    provider: str = getattr(settings, "EMBEDDING_PROVIDER", "ollama")
    model: str = getattr(settings, "EMBEDDING_MODEL", "nomic-embed-text")


class LLMFactory:
    @staticmethod
    def create_chat_model(cfg: LLMConfig | None = None):
        cfg = cfg or LLMConfig()
        if cfg.provider == "ollama":
            return ChatOllama(model=cfg.model, temperature=cfg.temperature)
        raise ValueError(f"Unsupported LLM provider: {cfg.provider}")

    @staticmethod
    def create_embedder(cfg: EmbeddingConfig | None = None):
        cfg = cfg or EmbeddingConfig()
        if cfg.provider == "ollama":
            return OllamaEmbeddings(model=cfg.model)
        raise ValueError(f"Unsupported Embedding provider: {cfg.provider}")
