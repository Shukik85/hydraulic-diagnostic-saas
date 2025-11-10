"""Pydantic схемы для RAG Service."""
from pydantic import BaseModel, Field

class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="User natural language question")
    context: dict | None = Field(default=None, description="Optional context object")
    equipment_id: str | None = Field(default=None, description="Equipment ID if specific")
    language: str = Field("ru", description="Language: ru/en")

class RAGQueryResponse(BaseModel):
    answer: str = Field(...)
    sources: list[str] = Field(...)
    score: float = Field(0.9)
    reasoning: str | None = Field(default=None)

class RAGAttentionExplainRequest(BaseModel):
    attention_weights: list[float] = Field(..., description="GNN attention scores per component")
    equipment_id: str | None = Field(default=None)
    gnn_reasoning: str | None = Field(default=None)
    language: str = Field("ru")

class RAGAttentionExplainResponse(BaseModel):
    explainer_text: str = Field(...)
    graph_snippet: str | None = Field(default=None)
    attention: list[float] = Field(...)

class RAGHistoryRequest(BaseModel):
    equipment_id: str
    since: str = Field(..., description="ISO8601 start date")
    until: str = Field(..., description="ISO8601 end date")
    max_docs: int = Field(10, description="Max docs to retrieve")
    language: str = Field("ru")

class RAGHistoryResponse(BaseModel):
    excerpt_text: str
    timeline_graph: str | None = Field(default=None)
    history_docs: list[dict] = Field([])
    confidence: float = Field(0.95)

class HealthResponse(BaseModel):
    status: str
    faiss: bool = Field(default=True)
    ollama: bool = Field(default=True)
    model_loaded: bool = Field(default=True)
