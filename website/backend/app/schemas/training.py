from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    project_id: str
    task_type: Literal["classification", "regression"]
    model_name: str
    target_column: str
    feature_columns: list[str] = Field(min_length=1)
    test_size: float = 0.2
    random_state: int = 42
    run_cv: bool = True
    use_hyperparameter_tuning: bool = False
    n_trials: int = Field(default=15, ge=2, le=100)
    pruning_enabled: bool = True
    tuning_metric: str | None = None


class TrainTaskResponse(BaseModel):
    task_id: str
    project_id: str
    status: str = "queued"
    message: str = "Model training task dispatched to background queue."


class TrainStatusResponse(BaseModel):
    task_id: str
    state: str
    status: str | None = None
    progress: int | None = None
    message: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None


class TelemetryMessage(BaseModel):
    task_id: str
    project_id: str
    status: str
    progress: int
    message: str
    timestamp: str | None = None
    snapshot: dict[str, object] | None = None
    error: str | None = None

