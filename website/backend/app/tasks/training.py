from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis

from app.core.celery_app import celery_app
from app.core.config import settings
from app.schemas.training import TrainRequest
from app.services.dataset_service import dataset_store
from app.services.training_service import training_service

logger = logging.getLogger(__name__)


# In-memory registry for local task tracking & fallback
task_state_store: dict[str, dict[str, Any]] = {}


def get_task_state(task_id: str) -> dict[str, Any] | None:
    return task_state_store.get(task_id)


def _get_redis_client() -> redis.Redis | None:
    try:
        r = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        r.ping()
        return r
    except Exception as exc:
        logger.warning("Redis not directly reachable for pub/sub: %s", exc)
        return None


def execute_training_direct(
    project_id: str,
    request_data: dict[str, Any],
    task_id: str = "task-local",
) -> dict[str, Any]:
    redis_client = _get_redis_client()
    channel = f"telemetry:{task_id}"

    task_state_store[task_id] = {
        "state": "PROGRESS",
        "info": {
            "task_id": task_id,
            "project_id": project_id,
            "status": "queued",
            "progress": 5,
            "message": "Task queued for worker...",
        },
    }

    def publish_telemetry(payload: dict[str, Any]) -> None:
        event = {
            "task_id": task_id,
            "project_id": project_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        task_state_store[task_id] = {"state": "PROGRESS", "info": event}
        if redis_client is not None:
            try:
                redis_client.publish(channel, json.dumps(event, default=str))
            except Exception as exc:
                logger.warning("Failed to publish telemetry to Redis: %s", exc)

    publish_telemetry({
        "status": "queued",
        "progress": 5,
        "message": "Task picked up by worker. Initializing training...",
    })

    try:
        session = dataset_store.get_project(project_id)
        train_req = TrainRequest(**request_data)
        updated_session = training_service.train(
            session,
            train_req,
            progress_callback=publish_telemetry,
        )
        snapshot = dataset_store.build_snapshot(updated_session)
        snapshot_dict = snapshot.model_dump()
        task_state_store[task_id] = {"state": "SUCCESS", "result": snapshot_dict}
        return snapshot_dict
    except Exception as exc:
        logger.exception("Error during model training task %s: %s", task_id, exc)
        task_state_store[task_id] = {"state": "FAILURE", "result": str(exc)}
        publish_telemetry({
            "status": "failed",
            "progress": 100,
            "error": str(exc),
            "message": f"Training failed: {exc}",
        })
        raise


@celery_app.task(bind=True, name="tasks.train_model")
def train_model_task(self, project_id: str, request_data: dict[str, Any]) -> dict[str, Any]:
    task_id = self.request.id or "task-local"
    return execute_training_direct(project_id, request_data, task_id=task_id)

