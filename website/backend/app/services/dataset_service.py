from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.results import DatasetSummary, ProjectSnapshot


import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("uploads")


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
    def __init__(self, base_dir: Path | None = None) -> None:
        self._projects: dict[str, ProjectSession] = {}
        self.base_dir = base_dir or UPLOADS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        p_dir = self.base_dir / project_id
        p_dir.mkdir(parents=True, exist_ok=True)
        return p_dir

    def save_project(self, session: ProjectSession) -> None:
        self._projects[session.project_id] = session
        p_dir = self._project_dir(session.project_id)
        try:
            # Save dataframes
            session.processed_data.to_parquet(p_dir / "processed.parquet")
            session.raw_data.to_parquet(p_dir / "raw.parquet")
        except Exception:
            session.processed_data.to_csv(p_dir / "processed.csv", index=False)
            session.raw_data.to_csv(p_dir / "raw.csv", index=False)

        meta = {
            "project_id": session.project_id,
            "input_kind": session.input_kind,
            "source_filename": session.source_filename,
            "source_mime_type": session.source_mime_type,
            "file_size_bytes": session.file_size_bytes,
            "preprocessing_log": session.preprocessing_log,
            "target_column": session.target_column,
            "feature_columns": session.feature_columns,
            "task_type": session.task_type,
            "artifact_path": session.artifact_path,
            "artifact_filename": session.artifact_filename,
            "artifact_skops_path": session.artifact_skops_path,
            "artifact_skops_filename": session.artifact_skops_filename,
            "artifact_onnx_path": session.artifact_onnx_path,
            "artifact_onnx_filename": session.artifact_onnx_filename,
            "model_results": session.model_results,
        }
        (p_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

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
        self.save_project(session)
        return session

    def get_project(self, project_id: str) -> ProjectSession:
        if project_id in self._projects:
            # Sync metadata from disk in case a worker updated it
            p_dir = self._project_dir(project_id)
            meta_path = p_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    session = self._projects[project_id]
                    session.target_column = meta.get("target_column")
                    session.feature_columns = meta.get("feature_columns", [])
                    session.task_type = meta.get("task_type")
                    session.artifact_path = meta.get("artifact_path")
                    session.artifact_filename = meta.get("artifact_filename")
                    session.artifact_skops_path = meta.get("artifact_skops_path")
                    session.artifact_skops_filename = meta.get("artifact_skops_filename")
                    session.artifact_onnx_path = meta.get("artifact_onnx_path")
                    session.artifact_onnx_filename = meta.get("artifact_onnx_filename")
                    session.model_results = meta.get("model_results")
                except Exception as exc:
                    logger.warning("Could not sync metadata from disk: %s", exc)
            return self._projects[project_id]

        p_dir = self._project_dir(project_id)
        meta_path = p_dir / "meta.json"
        if not meta_path.exists():
            raise KeyError(f"Unknown project id: {project_id}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (p_dir / "processed.parquet").exists():
            processed_df = pd.read_parquet(p_dir / "processed.parquet")
            raw_df = pd.read_parquet(p_dir / "raw.parquet")
        elif (p_dir / "processed.csv").exists():
            processed_df = pd.read_csv(p_dir / "processed.csv")
            raw_df = pd.read_csv(p_dir / "raw.csv")
        else:
            raise KeyError(f"Project data files not found for project: {project_id}")

        session = ProjectSession(
            project_id=project_id,
            raw_data=raw_df,
            processed_data=processed_df,
            input_kind=meta.get("input_kind", "csv"),
            source_filename=meta.get("source_filename"),
            source_mime_type=meta.get("source_mime_type"),
            file_size_bytes=meta.get("file_size_bytes"),
            preprocessing_log=meta.get("preprocessing_log", []),
            target_column=meta.get("target_column"),
            feature_columns=meta.get("feature_columns", []),
            task_type=meta.get("task_type"),
            artifact_path=meta.get("artifact_path"),
            artifact_filename=meta.get("artifact_filename"),
            artifact_skops_path=meta.get("artifact_skops_path"),
            artifact_skops_filename=meta.get("artifact_skops_filename"),
            artifact_onnx_path=meta.get("artifact_onnx_path"),
            artifact_onnx_filename=meta.get("artifact_onnx_filename"),
            model_results=meta.get("model_results"),
        )
        self._projects[project_id] = session
        return session

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
        self.save_project(session)

    def reset_to_raw(self, session: ProjectSession) -> ProjectSession:
        session.processed_data = session.raw_data.copy()
        session.preprocessing_log = []
        self.clear_model_state(session)
        self.save_project(session)
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
        optuna_res = None
        shap_res = None
        if session.model_results and isinstance(session.model_results, dict):
            optuna_res = session.model_results.get("optuna_optimization")
            shap_res = session.model_results.get("shap_explanations")

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
            optuna_results=optuna_res,
            shap_results=shap_res,
        )


dataset_store = DatasetStore()

