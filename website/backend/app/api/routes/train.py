from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

import redis.asyncio as aioredis
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, StreamingResponse

from app.core.celery_app import celery_app
from app.core.config import settings
from app.schemas.training import TrainRequest, TrainStatusResponse, TrainTaskResponse
from app.services.dataset_service import dataset_store
from app.services.visualization_service import visualization_service
from app.tasks.training import execute_training_direct, get_task_state, train_model_task


def _get_task_info(task_id: str) -> tuple[str, Any, Any]:
    # 1. Try checking local in-memory registry first
    local_state = get_task_state(task_id)
    if local_state is not None:
        return (
            local_state.get("state", "PENDING"),
            local_state.get("result"),
            local_state.get("info"),
        )

    # 2. Query Celery result backend
    try:
        res = AsyncResult(task_id, app=celery_app)
        state = res.state
        result = res.result if res.ready() else None
        info = res.info if state == "PROGRESS" else None
        return state, result, info
    except Exception as exc:
        logger.debug("Could not read Celery task meta: %s", exc)
        return "PENDING", None, None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["train"])


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


@router.post("", response_model=TrainTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model(request: TrainRequest) -> TrainTaskResponse:
    import io
    import pandas as pd

    # Ingest client-side preprocessed data if handed off from WebAssembly (Pyodide)
    if request.preprocessed_data is not None and len(request.preprocessed_data) > 0:
        df_handoff = pd.DataFrame(request.preprocessed_data)
        session = dataset_store.create_or_update_session_from_handoff(
            project_id=request.project_id,
            df=df_handoff,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            task_type=request.task_type,
            input_kind="pyodide_wasm",
        )
        dataset_store.clear_model_state(session)
    elif request.dataset_csv is not None and len(request.dataset_csv.strip()) > 0:
        df_handoff = pd.read_csv(io.StringIO(request.dataset_csv))
        session = dataset_store.create_or_update_session_from_handoff(
            project_id=request.project_id,
            df=df_handoff,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            task_type=request.task_type,
            input_kind="pyodide_wasm",
        )
        dataset_store.clear_model_state(session)
    else:
        try:
            session = dataset_store.get_project(request.project_id)
            dataset_store.clear_model_state(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    import uuid
    import redis
    task_id = uuid.uuid4().hex

    try:
        redis_available = False
        try:
            r = redis.Redis.from_url(
                settings.redis_url,
                socket_timeout=0.2,
                socket_connect_timeout=0.2,
            )
            r.ping()
            redis_available = True
        except Exception:
            redis_available = False

        if redis_available:
            task = train_model_task.apply_async(
                args=[request.project_id, request.model_dump()],
                task_id=task_id,
            )
            return TrainTaskResponse(
                task_id=task.id or task_id,
                project_id=request.project_id,
                status="queued",
                message=f"Model training task for {request.model_name} dispatched to Celery background queue.",
            )
        else:
            execute_training_direct(
                project_id=request.project_id,
                request_data=request.model_dump(),
                task_id=task_id,
            )
            return TrainTaskResponse(
                task_id=task_id,
                project_id=request.project_id,
                status="queued",
                message=f"Model training task for {request.model_name} executed in local worker mode.",
            )
    except Exception as exc:
        logger.exception("Failed to dispatch training task: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dispatch background training task: {exc}",
        ) from exc




@router.get("/status/{task_id}", response_model=TrainStatusResponse)
def get_train_status(task_id: str) -> TrainStatusResponse:
    state, result, info = _get_task_info(task_id)

    if state == "SUCCESS":
        return TrainStatusResponse(
            task_id=task_id,
            state=state,
            status="completed",
            progress=100,
            message="Model training completed successfully.",
            result=result if isinstance(result, dict) else None,
        )
    elif state == "PROGRESS":
        info_dict = info or {}
        return TrainStatusResponse(
            task_id=task_id,
            state=state,
            status=info_dict.get("status", "processing"),
            progress=info_dict.get("progress", 50),
            message=info_dict.get("message", "Training in progress..."),
        )
    elif state == "FAILURE":
        return TrainStatusResponse(
            task_id=task_id,
            state=state,
            status="failed",
            progress=100,
            error=str(result),
            message=f"Training task failed: {result}",
        )
    else:  # PENDING, RECEIVED, STARTED
        return TrainStatusResponse(
            task_id=task_id,
            state=state,
            status="queued",
            progress=5,
            message="Task is queued and waiting for an available worker...",
        )


@router.websocket("/ws/{task_id}")
async def websocket_training_telemetry(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    channel_name = f"telemetry:{task_id}"
    redis_client = None
    pubsub = None

    try:
        # 1. Send initial status
        state, result, info = _get_task_info(task_id)
        initial_event = {
            "task_id": task_id,
            "status": "connected",
            "progress": 5 if state == "PENDING" else (100 if state == "SUCCESS" else 50),
            "message": f"Connected to telemetry stream for task {task_id}.",
            "state": state,
        }
        if state == "SUCCESS" and isinstance(result, dict):
            initial_event["status"] = "completed"
            initial_event["progress"] = 100
            initial_event["snapshot"] = result
        await websocket.send_text(json.dumps(initial_event))

        if state in ("SUCCESS", "FAILURE"):
            try:
                while True:
                    await websocket.receive_text()
            except Exception:
                pass
            return

        # 2. Try subscribing to Redis
        try:
            redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_timeout=0.5,
                socket_connect_timeout=0.2,
            )
            pubsub = redis_client.pubsub()
            await asyncio.wait_for(pubsub.subscribe(channel_name), timeout=0.3)
        except Exception as exc:
            logger.info("Redis pubsub not available for WebSocket, using Celery polling fallback: %s", exc)
            pubsub = None

        while True:
            if pubsub is not None:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
                    if message and message["type"] == "message":
                        raw_data = message["data"]
                        await websocket.send_text(raw_data)
                        try:
                            parsed = json.loads(raw_data)
                            if parsed.get("status") in ("completed", "failed"):
                                break
                        except Exception:
                            pass
                except Exception:
                    pubsub = None

            # Check Celery task status
            curr_state, curr_result, curr_info = _get_task_info(task_id)
            if curr_state == "SUCCESS":
                completion_event = {
                    "task_id": task_id,
                    "status": "completed",
                    "progress": 100,
                    "message": "Model training completed successfully.",
                    "snapshot": curr_result if isinstance(curr_result, dict) else None,
                }
                await websocket.send_text(json.dumps(completion_event))
                try:
                    while True:
                        await websocket.receive_text()
                except Exception:
                    pass
                break
            elif curr_state == "FAILURE":
                failure_event = {
                    "task_id": task_id,
                    "status": "failed",
                    "progress": 100,
                    "error": str(curr_result),
                    "message": f"Training failed: {curr_result}",
                }
                await websocket.send_text(json.dumps(failure_event))
                try:
                    while True:
                        await websocket.receive_text()
                except Exception:
                    pass
                break
            elif curr_state == "PROGRESS" and curr_info:
                await websocket.send_text(json.dumps(curr_info))

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("Client disconnected from telemetry stream %s", task_id)
    except Exception as exc:
        logger.warning("Error in telemetry WebSocket for task %s: %s", task_id, exc)
    finally:
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(channel_name)
                await pubsub.close()
            except Exception:
                pass
        if redis_client is not None:
            try:
                await redis_client.close()
            except Exception:
                pass


@router.get("/telemetry/{task_id}")
async def sse_training_telemetry(task_id: str) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        channel_name = f"telemetry:{task_id}"
        redis_client = None
        pubsub = None
        try:
            try:
                redis_client = aioredis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_timeout=0.5,
                    socket_connect_timeout=0.2,
                )
                pubsub = redis_client.pubsub()
                await asyncio.wait_for(pubsub.subscribe(channel_name), timeout=0.3)
            except Exception:
                pubsub = None

            while True:
                if pubsub is not None:
                    try:
                        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
                        if message and message["type"] == "message":
                            yield f"data: {message['data']}\n\n"
                            try:
                                parsed = json.loads(message["data"])
                                if parsed.get("status") in ("completed", "failed"):
                                    break
                            except Exception:
                                pass
                    except Exception:
                        pubsub = None

                curr_state, curr_result, curr_info = _get_task_info(task_id)
                if curr_state == "SUCCESS":
                    completion_event = {
                        "task_id": task_id,
                        "status": "completed",
                        "progress": 100,
                        "message": "Model training completed successfully.",
                        "snapshot": curr_result if isinstance(curr_result, dict) else None,
                    }
                    yield f"data: {json.dumps(completion_event)}\n\n"
                    break
                elif curr_state == "FAILURE":
                    failure_event = {
                        "task_id": task_id,
                        "status": "failed",
                        "progress": 100,
                        "error": str(curr_result),
                    }
                    yield f"data: {json.dumps(failure_event)}\n\n"
                    break
                elif curr_state == "PROGRESS" and curr_info:
                    yield f"data: {json.dumps(curr_info)}\n\n"

                await asyncio.sleep(0.1)

        finally:
            if pubsub is not None:
                try:
                    await pubsub.unsubscribe(channel_name)
                    await pubsub.close()
                except Exception:
                    pass
            if redis_client is not None:
                try:
                    await redis_client.close()
                except Exception:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")




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


@router.get("/{project_id}/shap")
def shap_explanations(project_id: str) -> dict[str, object]:
    try:
        session = dataset_store.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    shap_data = session.model_results.get("shap_explanations") if session.model_results else None
    optuna_data = session.model_results.get("optuna_optimization") if session.model_results else None

    return {
        "project_id": project_id,
        "shap_explanations": shap_data,
        "optuna_optimization": optuna_data,
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
