from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.results import DatasetSummary, ProjectSnapshot


@dataclass
class ProjectSession:
    project_id: str
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame
    input_kind: str = "csv"
    source_filename: str | None = None
    source_mime_type: str | None = None
    file_size_bytes: int | None = None
    preprocessing_log: list[str] = field(default_factory=list)
    model_trainer: Any = None
    model_results: dict[str, Any] | None = None
    target_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    task_type: str | None = None
    artifact_path: str | None = None
    artifact_filename: str | None = None
    artifact_skops_path: str | None = None
    artifact_skops_filename: str | None = None
    artifact_onnx_path: str | None = None
    artifact_onnx_filename: str | None = None


class DatasetStore:
    def __init__(self) -> None:
        self._projects: dict[str, ProjectSession] = {}

    def create_project(
        self,
        df: pd.DataFrame,
        *,
        input_kind: str = "csv",
        source_filename: str | None = None,
        source_mime_type: str | None = None,
        file_size_bytes: int | None = None,
    ) -> ProjectSession:
        project_id = uuid.uuid4().hex
        session = ProjectSession(
            project_id=project_id,
            input_kind=input_kind,
            source_filename=source_filename,
            source_mime_type=source_mime_type,
            file_size_bytes=file_size_bytes,
            raw_data=df.copy(),
            processed_data=df.copy(),
        )
        self._projects[project_id] = session
        return session

    def get_project(self, project_id: str) -> ProjectSession:
        if project_id not in self._projects:
            raise KeyError(f"Unknown project id: {project_id}")
        return self._projects[project_id]

    def clear_model_state(self, session: ProjectSession) -> None:
        session.model_trainer = None
        session.model_results = None
        session.target_column = None
        session.feature_columns = []
        session.task_type = None
        session.artifact_path = None
        session.artifact_filename = None
        session.artifact_skops_path = None
        session.artifact_skops_filename = None
        session.artifact_onnx_path = None
        session.artifact_onnx_filename = None

    def reset_to_raw(self, session: ProjectSession) -> ProjectSession:
        session.processed_data = session.raw_data.copy()
        session.preprocessing_log = []
        self.clear_model_state(session)
        return session

    def build_summary(self, session: ProjectSession) -> DatasetSummary:
        df = session.processed_data
        preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")
        return DatasetSummary(
            project_id=session.project_id,
            input_kind=session.input_kind,
            source_filename=session.source_filename,
            source_mime_type=session.source_mime_type,
            file_size_bytes=session.file_size_bytes,
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            column_names=[str(name) for name in df.columns.tolist()],
            missing_values=int(df.isnull().sum().sum()),
            duplicate_rows=int(df.duplicated().sum()),
            preview=preview,
            preprocessing_log=session.preprocessing_log.copy(),
        )

    def build_snapshot(self, session: ProjectSession) -> ProjectSnapshot:
        has_skops = bool(session.artifact_skops_path)
        has_onnx = bool(session.artifact_onnx_path)
        return ProjectSnapshot(
            summary=self.build_summary(session),
            target_column=session.target_column,
            feature_columns=session.feature_columns.copy(),
            model_results=session.model_results,
            trained_task_type=session.task_type,
            artifact_available=bool(session.artifact_path or has_skops or has_onnx),
            artifact_filename=session.artifact_skops_filename or session.artifact_filename,
            skops_artifact_available=has_skops,
            onnx_artifact_available=has_onnx,
            skops_artifact_filename=session.artifact_skops_filename,
            onnx_artifact_filename=session.artifact_onnx_filename,
        )


dataset_store = DatasetStore()
