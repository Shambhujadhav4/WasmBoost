from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    project_id: str
    input_kind: str = "csv"
    source_filename: str | None = None
    source_mime_type: str | None = None
    file_size_bytes: int | None = None
    rows: int
    columns: int
    column_names: list[str]
    missing_values: int
    duplicate_rows: int
    preview: list[dict[str, Any]] = Field(default_factory=list)
    preprocessing_log: list[str] = Field(default_factory=list)


class ColumnInfoRow(BaseModel):
    column: str
    dtype: str
    non_null: int
    null: int
    unique: int


class DatasetInsights(BaseModel):
    project_id: str
    summary: DatasetSummary
    column_info: list[ColumnInfoRow] = Field(default_factory=list)
    descriptive_statistics: list[dict[str, Any]] = Field(default_factory=list)


class ProjectSnapshot(BaseModel):
    summary: DatasetSummary
    target_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    model_results: dict[str, Any] | None = None
    trained_task_type: str | None = None
    artifact_available: bool = False
    artifact_filename: str | None = None
    skops_artifact_available: bool = False
    onnx_artifact_available: bool = False
    skops_artifact_filename: str | None = None
    onnx_artifact_filename: str | None = None


class ModelRecommendation(BaseModel):
    model_name: str
    mean_score: float
    std_score: float
    metric_scores: dict[str, float] = Field(default_factory=dict)


class WorkflowRecommendation(BaseModel):
    project_id: str
    recommended_task_type: str | None = None
    recommended_target_column: str | None = None
    benchmark_metric: str | None = None
    available_benchmark_metrics: list[str] = Field(default_factory=list)
    best_model: ModelRecommendation | None = None
    candidate_models: list[ModelRecommendation] = Field(default_factory=list)
    suggested_preprocessing_steps: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
