"""Модуль проекта с автогенерированным докстрингом."""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import django
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import numpy as np

# Инициализация Django окружения (для доступа к settings, если нужно)
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

django.setup()

# После django.setup() можно безопасно импортировать Django-зависимые модули
from apps.rag_assistant.llm_factory import LLMFactory  # noqa: E402
from apps.rag_assistant.rag_core import (  # noqa: E402
    DEFAULT_LOCAL_STORAGE,
    LocalStorageBackend,
    VectorIndex,
)


def format_docs(docs: list[dict[str, Any]]) -> str:
    """Выполняет format docs

    pass
    Args:
        docs (Any): Параметр docs

    """
    return "\n\n".join(d["content"] for d in docs if d.get("content"))


def build_rag_chain(vindex: VectorIndex, ollama_embedder, llm):
    """Выполняет build rag chain

    pass
    Args:
        vindex (Any): Параметр vindex
        ollama_embedder (Any): Параметр ollama_embedder
        llm (Any): Параметр llm

    """

    def _encode(texts: list[str]) -> np.ndarray:
        """Выполняет  encode

        Args:
            texts (Any): Параметр texts

        """
        embs = ollama_embedder.embed_documents(texts)
        arr = np.array(embs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype("float32")

    docs_list = (vindex.metadata or {}).get("docs", [])

    def retrieve(question: str):
        """Выполняет retrieve

        Args:
            question (Any): Параметр question

        """
        q_emb = _encode([question])
        _, indices = vindex.search(q_emb, k=4)
        results = []
        for i in indices[0]:
            if 0 <= i < len(docs_list):
                results.append({"content": docs_list[i], "meta": {"idx": int(i)}})
        return results

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant for hydraulic diagnostics.
        Use the following retrieved context to answer the user's question concisely and accurately.
        If the answer is not present in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}

        Answer in the same language as the question.
        """
    )
    parser = StrOutputParser()

    return (
        {"question": RunnablePassthrough()}
        | {"docs": RunnableLambda(lambda x: retrieve(x["question"]))}
        | RunnableLambda(
            lambda x: {"question": x["question"], "context": format_docs(x["docs"])}
        )
        | prompt
        | llm
        | parser
    )


def main():
    """Выполняет main"""
    print("✅ AI Engine и RAG система инициализированы")

    # 1) Исходные документы
    docs = [
        "Hydraulic pressure is low, check pump operation and fluid levels",
        "Pressure relief valve stuck open, system cannot maintain pressure",
        "Pump failure suspected due to unusual noise and vibration patterns",
        "Filter clogged, causing cavitation and reduced flow",
        "Air in hydraulic lines can lead to spongy response and delayed actuation",
    ]
    version = "v_test_v2"

    # 2) Сборка индекса: embeddings через Ollama для совместимости с retriever
    print("📦 Building embeddings and index via Ollama (nomic-embed-text)...")
    storage = LocalStorageBackend(base_path=Path(DEFAULT_LOCAL_STORAGE).absolute())

    # Сгенерировать эмбеддинги через OllamaEmbeddings
    ollama_embedder = LLMFactory.create_embedder()  # nomic-embed-text
    embs = np.array(ollama_embedder.embed_documents(docs), dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = (embs / norms).astype("float32")

    vindex = VectorIndex(dim=embs.shape[1], metric="ip")
    vindex.build(embs)
    index_bytes = vindex.to_bytes()
    meta = {"dim": embs.shape[1], "metric": "ip", "docs": docs}
    save_path = storage.save_index(version, index_bytes, meta)
    print(json.dumps({"event": "index_saved", "path": save_path}, ensure_ascii=False))

    # 3) Загрузка индекса и сборка цепочки
    idx_bytes, loaded_meta = storage.load_index(version)
    vindex2 = VectorIndex.from_bytes(idx_bytes)
    vindex2.metadata = loaded_meta

    llm = LLMFactory.create_chat_model()  # qwen3:8b по умолчанию

    chain = build_rag_chain(vindex2, ollama_embedder, llm)

    # 4) Запросы
    print("🔍 Testing RAG chain...")
    queries = ["pump problems", "pressure issues", "air in lines"]
    for _i, q in enumerate(queries, 1):
        t0 = time.time()
        answer = chain.invoke({"question": q})
        dt = time.time() - t0
        payload = {"query": q, "answer": answer, "t_sec": round(dt, 2)}
        print(json.dumps(payload, ensure_ascii=False))

    print("\n🎉 RAG test completed successfully!")


if __name__ == "__main__":
    main()
