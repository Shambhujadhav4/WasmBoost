from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DropColumnsRequest(BaseModel):
    project_id: str
    columns: list[str] = Field(min_length=1)


class MissingValuesRequest(BaseModel):
    project_id: str
    strategy: Literal["drop_rows", "drop_columns", "mean", "median", "mode", "zero", "custom"]
    columns: list[str] = Field(default_factory=list)
    fill_value: str | float | int | None = None


class EncodeRequest(BaseModel):
    project_id: str
    method: Literal["label", "onehot"]
    columns: list[str] = Field(default_factory=list)


class ScaleRequest(BaseModel):
    project_id: str
    method: Literal["standard", "minmax", "robust"]
    columns: list[str] = Field(default_factory=list)


class OutlierRequest(BaseModel):
    project_id: str
    method: Literal["iqr_remove", "iqr_cap", "zscore_remove"]
    columns: list[str] = Field(default_factory=list)
    threshold: float = 1.5


class MutualInformationRequest(BaseModel):
    project_id: str
    target_column: str
    feature_columns: list[str] = Field(default_factory=list)
    task_type: Literal["classification", "regression"] | None = None


class MutualInformationItem(BaseModel):
    feature: str
    score: float
    normalized_score: float


class MutualInformationResponse(BaseModel):
    project_id: str
    target_column: str
    task_type: str
    scores: list[MutualInformationItem] = Field(default_factory=list)
    figure: dict[str, object] | None = None


class FeatureSelectionRequest(BaseModel):
    project_id: str
    method: Literal["mi", "rfe"]
    target_column: str
    n_features_to_select: int = Field(default=5, ge=1)
    task_type: Literal["classification", "regression"] | None = None
    feature_columns: list[str] = Field(default_factory=list)
    rfe_estimator: str | None = "Random Forest"
    step: float = Field(default=1.0, gt=0)


class FeatureRankingItem(BaseModel):
    feature: str
    ranking: int
    selected: bool
    score: float | None = None


class FeatureSelectionResponse(BaseModel):
    project_id: str
    method: str
    target_column: str
    selected_features: list[str]
    eliminated_features: list[str]
    rankings: list[FeatureRankingItem] = Field(default_factory=list)
    snapshot: dict[str, object] | None = None
