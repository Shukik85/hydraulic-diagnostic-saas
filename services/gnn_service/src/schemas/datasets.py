"""Dataset ingestion schemas.

API rule (single-version API):
- mapping_json is required
- provide exactly one of topology_id or topology_json

Two ingestion modes:
- /datasets/ingest (multipart upload) -> handled in endpoint params
- /datasets/ingest-by-url (JSON body) -> validated by these schemas
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, model_validator


class DatasetIngestByUrlRequest(BaseModel):
    """Ingest dataset by URL.

    dataset_url should be a presigned URL or any http(s) URL accessible by the service.
    """

    equipment_id: str | None = Field(
        default=None,
        description="Optional equipment identifier (tenant/user-level).",
        examples=["zema_001"],
    )

    dataset_url: HttpUrl = Field(
        ...,
        description="Dataset file URL (Parquet/CSV).",
    )

    mapping_json: dict[str, Any] = Field(
        ...,
        description="Mapping JSON payload.",
    )

    mapping_url: HttpUrl | None = Field(
        default=None,
        description="Optional URL to mapping JSON (if you don't want to send mapping_json inline).",
    )

    topology_id: str | None = Field(
        default=None,
        description="Topology identifier (use this OR topology_json).",
    )

    topology_json: dict[str, Any] | None = Field(
        default=None,
        description="Topology payload (use this OR topology_id).",
    )

    @model_validator(mode="after")
    def _validate_topology_one_of(self) -> "DatasetIngestByUrlRequest":
        if bool(self.topology_id) == bool(self.topology_json):
            raise ValueError("Provide exactly one: topology_id or topology_json")
        return self

    @model_validator(mode="after")
    def _validate_mapping(self) -> "DatasetIngestByUrlRequest":
        if not isinstance(self.mapping_json, dict) or not self.mapping_json:
            raise ValueError("mapping_json must be a non-empty JSON object")
        return self
