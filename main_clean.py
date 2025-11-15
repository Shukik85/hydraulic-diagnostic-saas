"""RAG Service - Minimal Clean Version"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="RAG Service",
    version="2.0.0",
    description="Hydraulic Diagnostic RAG API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "RAG Service is running", "version": "2.0.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "rag",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
