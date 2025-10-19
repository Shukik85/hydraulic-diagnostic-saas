import os
import sys
import django
import time
from typing import List, Dict, Any
import numpy as np

from apps.rag_assistant.rag_core import RAGOrchestrator, LocalStorageBackend, EmbeddingsProvider, DEFAULT_LOCAL_STORAGE
from apps.rag_assistant.llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Инициализация Django окружения, если требуется доступ к settings
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

django.setup()


def format_docs(docs: List[Dict[str, Any]]) -> str:
    return "\n\n".join(d["content"] for d in docs if d.get("content"))


def build_rag_chain(vindex, embedder, llm):
    def _encode(texts: List[str]) -> np.ndarray:
        embs = embedder.embed_documents(texts)
        arr = np.array(embs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype("float32")

    docs_list = (vindex.metadata or {}).get("docs", [])

    def retrieve(question: str):
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

    chain = (
        {"question": RunnablePassthrough()}
        | {"docs": RunnableLambda(lambda x: retrieve(x["question"]))}
        | RunnableLambda(lambda x: {"question": x["question"], "context": format_docs(x["docs"])})
        | prompt
        | llm
        | parser
    )
    return chain


def main():
    print("✅ AI Engine и RAG система инициализированы")

    # 1) Данные (можете заменить на свои)
    docs = [
        "Hydraulic pressure is low, check pump operation and fluid levels",
        "Pressure relief valve stuck open, system cannot maintain pressure",
        "Pump failure suspected due to unusual noise and vibration patterns",
        "Filter clogged, causing cavitation and reduced flow",
        "Air in hydraulic lines can lead to spongy response and delayed actuation",
    ]
    version = "v_test_v2"

    # 2) Сборка индекса (сохранение docs в metadata)
    print("📦 Building embeddings and index via Ollama (nomic-embed-text)...")
    storage = LocalStorageBackend(base_path=os.path.abspath(DEFAULT_LOCAL_STORAGE))
    # В этом тесте embedder rag_core не используем — вместо него Ollama embeddings из LLMFactory
    # но orchestrator требует EmbeddingsProvider, поэтому подставим любой (он не будет использоваться ниже)
    dummy_embedder = EmbeddingsProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
    orchestrator = RAGOrchestrator(storage=storage, embedder=dummy_embedder, index_metric="ip")
    orchestrator()

    # Векторизацию выполним через OllamaEmbeddings, чтобы индекс соответствовал ретриверу
    ollama_embedder = LLMFactory.create_embedder()
    embs = np.array(ollama_embedder.embed_documents(docs), dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = (embs / norms).astype("float32")

    # Соберем индекс вручную, затем сохраним через orchestrator.save_index
    from apps.rag_assistant.rag_core import VectorIndex
    vindex = VectorIndex(dim=embs.shape[1], metric="ip")
    vindex.build(embs)
    index_bytes = vindex.to_bytes()
    meta = {"dim": embs.shape[1], "metric": "ip", "docs": docs}
    storage.save_index(version, index_bytes, meta)

    print(f"✅ Index saved to: {os.path.join(DEFAULT_LOCAL_STORAGE, f'v_{version}')}")

    # 3) Загрузка индекса и сборка цепочки
    idx_bytes, loaded_meta = storage.load_index(version)
    vindex2 = VectorIndex.from_bytes(idx_bytes)
    vindex2.metadata = loaded_meta

    llm = LLMFactory.create_chat_model()  # qwen3:8b по умолчанию

    chain = build_rag_chain(vindex2, ollama_embedder, llm)

    # 4) Запросы
    print("🔍 Testing RAG chain...")
    queries = ["pump problems", "pressure issues", "air in lines"]
    for i, q in enumerate(queries, 1):
        t0 = time.time()
        answer = chain.invoke({"question": q})
        dt = time.time() - t0
        print(f"\nQuery {i} - '{q}':")
        print("Answer:", answer)
        print(f"(took {dt:.2f}s)")

    print("\n🎉 RAG test completed successfully!")


if __name__ == "__main__":
    main()
