"""Pydantic schemas для API контрактов."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request для single equipment inference."""
    
    node_features: list[list[float]] = Field(
        ...,
        description="Node feature matrix (num_nodes, num_features)",
        examples=[[[185.0, 1780, 68.0], [160.0, 1650, 72.0]]],
    )
    edge_index: list[list[int]] = Field(
        ...,
        description="Edge connectivity (2, num_edges)",
        examples=[[[0, 1, 2], [1, 2, 3]]],
    )
    edge_attr: list[list[float]] | None = Field(
        default=None,
        description="Edge attributes (num_edges, edge_dim)",
        examples=[[[50, 180, 5.2], [45, 165, 4.8]]],
    )
    component_names: list[str] = Field(
        ...,
        description="Component names для explainability",
        examples=[["pump", "boom", "stick", "bucket"]],
    )


class ExplanationDetail(BaseModel):
    """Explainability details для attention analysis."""
    
    critical_components: list[str] = Field(
        ...,
        description="Components с высоким attention score",
    )
    attention_scores: list[float] = Field(
        ...,
        description="Attention scores per node",
    )
    causal_path: list[str] = Field(
        default_factory=list,
        description="Causal chain (e.g., pump_overheating → pressure_drop)",
    )
    reasoning: str = Field(
        ...,
        description="Human-readable reasoning",
    )


class PredictionResponse(BaseModel):
    """Response для single inference."""
    
    prediction: int = Field(
        ...,
        description="0 = NORMAL, 1 = ANOMALY",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score",
    )
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anomaly intensity",
    )
    explanation: ExplanationDetail | None = Field(
        default=None,
        description="Explainability (only if anomaly detected)",
    )


class BatchPredictionRequest(BaseModel):
    """Request для batch inference."""
    
    graphs: list[PredictionRequest] = Field(
        ...,
        description="List of graphs for fleet",
    )


class BatchPredictionResponse(BaseModel):
    """Response для batch inference."""
    
    predictions: list[dict] = Field(
        ...,
        description="List of predictions",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        ...,
        description="Service status",
    )
    model_loaded: bool = Field(
        ...,
        description="Model loaded flag",
    )
    device: str = Field(
        ...,
        description="Compute device (cuda/cpu)",
    )
