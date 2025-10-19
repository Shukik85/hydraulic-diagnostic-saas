SHELL := /bin/bash

BACKEND_DIR := backend
DJANGO_SETTINGS := core.settings
PY := python

rag-test:
	@echo "==> Running RAG smoke test with Ollama"
	cd $(BACKEND_DIR) && \
		env DJANGO_SETTINGS_MODULE=$(DJANGO_SETTINGS) \
		LLM_PROVIDER=ollama \
		LLM_MODEL=qwen3:8b \
		EMBEDDING_PROVIDER=ollama \
		EMBEDDING_MODEL=nomic-embed-text \
		$(PY) test_rag.py

rag-build-index:
	@echo "==> Building FAISS index with Ollama embeddings"
	cd $(BACKEND_DIR) && \
		env DJANGO_SETTINGS_MODULE=$(DJANGO_SETTINGS) \
		LLM_PROVIDER=ollama \
		EMBEDDING_PROVIDER=ollama \
		EMBEDDING_MODEL=nomic-embed-text \
		$(PY) -c "import os, numpy as np; from apps.rag_assistant.rag_core import LocalStorageBackend, DEFAULT_LOCAL_STORAGE, VectorIndex; from apps.rag_assistant.llm_factory import LLMFactory; docs = ['Hydraulic pressure is low, check pump and fluid levels', 'Relief valve stuck open, cannot maintain pressure', 'Pump failure suspected due to noise and vibration']; embedder = LLMFactory.create_embedder(); embs = np.array(embedder.embed_documents(docs), dtype='float32'); norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12; embs = (embs / norms).astype('float32'); vindex = VectorIndex(dim=embs.shape[1], metric='ip'); vindex.build(embs); idx_bytes = vindex.to_bytes(); storage = LocalStorageBackend(base_path=os.path.abspath(DEFAULT_LOCAL_STORAGE)); storage.save_index('make_idx', idx_bytes, {'docs': docs, 'dim': embs.shape[1], 'metric': 'ip'}); print('Index saved to:', os.path.join(DEFAULT_LOCAL_STORAGE, 'v_make_idx'))"

rag-clean:
	@echo "==> Cleaning local indexes (data/indexes)"
	rm -rf data/indexes || true

rag-ci-fallback:
	@echo "==> CI fallback: building FAISS index with sentence-transformers"
	cd $(BACKEND_DIR) && \
		env DJANGO_SETTINGS_MODULE=$(DJANGO_SETTINGS) \
		$(PY) -c "import os, numpy as np; from apps.rag_assistant.rag_core import LocalStorageBackend, DEFAULT_LOCAL_STORAGE, VectorIndex; from sentence_transformers import SentenceTransformer; docs = ['Hydraulic pressure is low, check pump and fluid levels', 'Relief valve stuck open, cannot maintain pressure', 'Pump failure suspected due to noise and vibration']; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); embs = model.encode(docs, convert_to_numpy=True).astype('float32'); norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12; embs = (embs / norms).astype('float32'); vindex = VectorIndex(dim=embs.shape[1], metric='ip'); vindex.build(embs); idx_bytes = vindex.to_bytes(); storage = LocalStorageBackend(base_path=os.path.abspath(DEFAULT_LOCAL_STORAGE)); storage.save_index('ci_fallback', idx_bytes, {'docs': docs, 'dim': embs.shape[1], 'metric': 'ip'}); print('Index saved to:', os.path.join(DEFAULT_LOCAL_STORAGE, 'v_ci_fallback'))"
