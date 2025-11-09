FROM hydraulic-ai-base-gpu:cuda12.1
LABEL maintainer="Aleksandr Plotnikov <a.s.plotnikov85@gmail.com>"
LABEL description="RAG Service (FAISS + DeepSeek-R1 via Ollama)"
WORKDIR /app
# System & python deps
RUN apt-get update && apt-get install -y --no-install-recommends git curl build-essential
# Install requirements
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY requirements-rag.txt /app/
RUN pip install -r requirements-rag.txt
# Copy code (app structure)
COPY app /app/app
COPY .env.example /app/.env.example
COPY README.md /app/README.md
# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8002/health || exit 1
# Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
