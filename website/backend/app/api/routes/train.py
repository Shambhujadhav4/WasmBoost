from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.results import ProjectSnapshot
from app.schemas.training import TrainRequest
from app.services.dataset_service import dataset_store
from app.services.training_service import training_service
from app.services.visualization_service import visualization_service


router = APIRouter(prefix="/train", tags=["train"])


from typing import Literal

from app.core.executor import run_in_process


def _get_artifact_file(session, format_type: str) -> tuple[Path, str]:
    normalized_format = (format_type or "skops").lower().strip()
    if normalized_format in {"onnx", ".onnx"}:
        if not session.artifact_onnx_path:
            raise HTTPException(
                status_code=404,
                detail="No ONNX model artifact is available for this project.",
            )
        artifact_path = Path(session.artifact_onnx_path)
        if not artifact_path.exists():
            raise HTTPException(
                status_code=404,
                detail="The ONNX model artifact could not be found on disk.",
            )
        download_name = session.artifact_onnx_filename or artifact_path.name
        return artifact_path, download_name

    # Default to secure .skops format
    path_str = session.artifact_skops_path or session.artifact_path
    if not path_str:
        raise HTTPException(
            status_code=404,
            detail="No saved model artifact is available for this project.",
        )
    artifact_path = Path(path_str)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail="The saved model artifact could not be found on disk.",
        )
    download_name = (
        session.artifact_skops_filename
        or session.artifact_filename
        or artifact_path.name
    )
    return artifact_path, download_name


@router.post("", response_model=ProjectSnapshot)
async def train_model(request: TrainRequest) -> ProjectSnapshot:
    try:
        session = dataset_store.get_project(request.project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        # Run training in a background thread so it doesn't block FastAPI
        session = await run_in_process(training_service.train, session, request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return dataset_store.build_snapshot(session)


@router.get("/{project_id}/feature-importance")
def feature_importance(project_id: str) -> dict[str, object]:
    try:
        session = dataset_store.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "project_id": project_id,
        "feature_importance": visualization_service.feature_importance(session),
    }


@router.get("/{project_id}/artifact")
def download_artifact(
    project_id: str,
    format: Literal["skops", "onnx"] = "skops",
) -> FileResponse:
    try:
        session = dataset_store.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    artifact_path, download_name = _get_artifact_file(session, format)
    return FileResponse(
        path=artifact_path,
        media_type="application/octet-stream",
        filename=download_name,
    )


@router.get("/{project_id}/artifact/{artifact_format}")
def download_artifact_by_format(
    project_id: str,
    artifact_format: str,
) -> FileResponse:
    try:
        session = dataset_store.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    artifact_path, download_name = _get_artifact_file(session, artifact_format)
    return FileResponse(
        path=artifact_path,
        media_type="application/octet-stream",
        filename=download_name,
    )
