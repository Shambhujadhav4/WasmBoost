from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from app.schemas.training import TrainRequest
from app.services.dataset_service import ProjectSession
from modules.models import ModelTrainer

logger = logging.getLogger(__name__)


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class TrainingService:
    def _artifact_dir(self, session: ProjectSession) -> Path:
        # Store in the persistent models directory relative to project root
        # In Docker this maps to /app/models
        artifact_dir = Path("models") / session.project_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _safe_model_name(self, session: ProjectSession) -> str:
        model_name = session.model_trainer.model_name if session.model_trainer else "model"
        return "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in (model_name or "model")
        ).strip("-") or "model"

    def _save_skops_artifact(self, session: ProjectSession, artifact_dir: Path, safe_model_name: str) -> Path:
        skops_path = artifact_dir / f"{safe_model_name}.skops"
        payload = {
            "project_id": session.project_id,
            "input_kind": session.input_kind,
            "source_filename": session.source_filename,
            "source_mime_type": session.source_mime_type,
            "task_type": session.task_type,
            "model_name": session.model_trainer.model_name,
            "feature_names": session.model_trainer.feature_names,
            "target_column": session.target_column,
            "model": session.model_trainer.model,
        }
        sio.dump(payload, skops_path)
        return skops_path

    def _save_onnx_artifact(self, session: ProjectSession, artifact_dir: Path, safe_model_name: str) -> Path | None:
        onnx_path = artifact_dir / f"{safe_model_name}.onnx"
        try:
            num_features = len(session.model_trainer.feature_names)
            initial_type = [("float_input", FloatTensorType([None, num_features]))]
            options: dict[Any, Any] = {}
            if session.task_type == "classification":
                options = {type(session.model_trainer.model): {"zipmap": False}}

            onx = to_onnx(
                session.model_trainer.model,
                initial_types=initial_type,
                options=options,
                target_opset=15,
            )
            onnx_path.write_bytes(onx.SerializeToString())
            return onnx_path
        except Exception as exc:
            logger.exception("Failed to convert model to ONNX: %s", exc)
            return None

    def _save_artifact(self, session: ProjectSession) -> None:
        if session.model_trainer is None or session.model_trainer.model is None:
            return

        artifact_dir = self._artifact_dir(session)
        safe_model_name = self._safe_model_name(session)

        # 1. Save secure Python model (.skops)
        skops_path = self._save_skops_artifact(session, artifact_dir, safe_model_name)
        session.artifact_skops_path = str(skops_path)
        session.artifact_skops_filename = skops_path.name

        # 2. Save production inference model (.onnx)
        onnx_path = self._save_onnx_artifact(session, artifact_dir, safe_model_name)
        if onnx_path is not None and onnx_path.exists():
            session.artifact_onnx_path = str(onnx_path)
            session.artifact_onnx_filename = onnx_path.name
        else:
            session.artifact_onnx_path = None
            session.artifact_onnx_filename = None

        # Keep backward compatibility fields pointing to the secure .skops artifact
        session.artifact_path = str(skops_path)
        session.artifact_filename = skops_path.name

    def train(self, session: ProjectSession, request: TrainRequest) -> ProjectSession:
        trainer = ModelTrainer()
        trainer.prepare_data(
            session.processed_data,
            request.target_column,
            request.feature_columns,
            request.test_size,
            request.random_state,
            request.task_type,
        )
        trainer.train(request.task_type, request.model_name)
        metrics = trainer.get_metrics()
        if request.run_cv:
            metrics["cv_scores"] = trainer.get_cross_val_scores(cv=5)

        session.model_trainer = trainer
        session.model_results = _serialize(metrics)
        session.target_column = request.target_column
        session.feature_columns = request.feature_columns.copy()
        session.task_type = request.task_type
        self._save_artifact(session)
        return session


training_service = TrainingService()
