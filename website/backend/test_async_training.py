"""Comprehensive test suite for Phase 3: Asynchronous Model Training with Celery & WebSockets.

Tests:
1. Celery Task Queue execution (`train_model_task`) with live telemetry callbacks.
2. FastAPI `POST /api/train` asynchronous dispatch (HTTP 202 Accepted).
3. FastAPI `GET /api/train/status/{task_id}` status polling.
4. FastAPI `WebSocket /api/train/ws/{task_id}` live telemetry stream.
5. Multi-framework model training (.skops and .onnx artifact creation in background).
6. Artifact download endpoints for async-trained models.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.core.celery_app import celery_app
from app.main import app
from app.schemas.training import TrainRequest
from app.services.dataset_service import dataset_store
from app.tasks.training import train_model_task


celery_app.conf.task_always_eager = True
celery_app.conf.result_backend = "cache+memory://"
celery_app.conf.task_store_eager_result = True


def create_sample_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n) * 2.0,
        "feature_3": np.random.randint(0, 5, size=n),
        "target": np.random.choice(["Class_A", "Class_B"], size=n),
    })


def test_celery_task_direct_execution() -> None:
    print("\n=== 1. Testing Celery train_model_task Direct Execution ===")
    df = create_sample_dataset()
    session = dataset_store.create_project(df, input_kind="csv", source_filename="async_test.csv")

    req_data = {
        "project_id": session.project_id,
        "task_type": "classification",
        "model_name": "Random Forest",
        "target_column": "target",
        "feature_columns": ["feature_1", "feature_2", "feature_3"],
        "test_size": 0.2,
        "random_state": 42,
        "run_cv": True,
    }

    task_result = train_model_task.apply(args=[session.project_id, req_data])
    assert task_result.successful(), f"Task failed: {task_result.result}"
    snapshot = task_result.result
    assert snapshot is not None
    assert snapshot["model_results"] is not None
    assert "accuracy" in snapshot["model_results"]
    assert "cv_scores" in snapshot["model_results"]
    assert snapshot["skops_artifact_available"] is True
    assert snapshot["onnx_artifact_available"] is True
    print(f"[OK] Task completed successfully. Accuracy: {snapshot['model_results']['accuracy']:.4f}")
    print(f"[OK] Skops artifact: {snapshot['skops_artifact_filename']}, ONNX artifact: {snapshot['onnx_artifact_filename']}")


def test_api_async_dispatch_and_status() -> None:
    print("\n=== 2. Testing API POST /api/train (HTTP 202) & Status Polling ===")
    client = TestClient(app)
    df = create_sample_dataset()
    session = dataset_store.create_project(df, input_kind="csv", source_filename="api_async_test.csv")

    payload = {
        "project_id": session.project_id,
        "task_type": "classification",
        "model_name": "XGBoost",
        "target_column": "target",
        "feature_columns": ["feature_1", "feature_2", "feature_3"],
        "test_size": 0.2,
        "random_state": 42,
        "run_cv": True,
    }

    # 1. POST /api/train -> HTTP 202 Accepted
    response = client.post("/api/train", json=payload)
    assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
    data = response.json()
    assert "task_id" in data
    assert data["project_id"] == session.project_id
    assert data["status"] == "queued"
    task_id = data["task_id"]
    print(f"[OK] POST /api/train returned HTTP 202 Accepted with task_id: {task_id}")

    # 2. GET /api/train/status/{task_id}
    status_resp = client.get(f"/api/train/status/{task_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["task_id"] == task_id
    assert status_data["state"] in ("SUCCESS", "PROGRESS", "PENDING")
    print(f"[OK] GET /api/train/status returned state: {status_data['state']}, status: {status_data.get('status')}")


def test_websocket_telemetry_stream() -> None:
    print("\n=== 3. Testing WebSocket Telemetry Endpoint (/api/train/ws/{task_id}) ===")
    client = TestClient(app)
    df = create_sample_dataset()
    session = dataset_store.create_project(df, input_kind="csv", source_filename="ws_test.csv")

    # Dispatch task
    payload = {
        "project_id": session.project_id,
        "task_type": "classification",
        "model_name": "LightGBM",
        "target_column": "target",
        "feature_columns": ["feature_1", "feature_2", "feature_3"],
        "test_size": 0.2,
        "random_state": 42,
        "run_cv": True,
    }
    resp = client.post("/api/train", json=payload)
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    # Connect to WebSocket
    with client.websocket_connect(f"/api/train/ws/{task_id}") as websocket:
        initial_msg = websocket.receive_text()
        parsed = json.loads(initial_msg)
        assert parsed["task_id"] == task_id
        assert parsed["status"] in ("connected", "completed")
        print(f"[OK] Received WebSocket telemetry event: status={parsed.get('status')}, progress={parsed.get('progress')}%")



def test_async_artifact_download() -> None:
    print("\n=== 4. Testing Artifact Download after Background Training ===")
    client = TestClient(app)
    df = create_sample_dataset()
    session = dataset_store.create_project(df, input_kind="csv", source_filename="download_test.csv")

    celery_app.conf.task_always_eager = True

    payload = {
        "project_id": session.project_id,
        "task_type": "classification",
        "model_name": "CatBoost",
        "target_column": "target",
        "feature_columns": ["feature_1", "feature_2", "feature_3"],
        "test_size": 0.2,
        "random_state": 42,
        "run_cv": True,
    }
    resp = client.post("/api/train", json=payload)
    assert resp.status_code == 202

    # Verify session has artifacts
    updated_session = dataset_store.get_project(session.project_id)
    assert updated_session.artifact_skops_path is not None
    assert updated_session.artifact_onnx_path is not None

    # Test downloading .skops
    skops_resp = client.get(f"/api/train/{session.project_id}/artifact?format=skops")
    assert skops_resp.status_code == 200
    assert len(skops_resp.content) > 0
    print(f"[OK] Downloaded .skops artifact ({len(skops_resp.content)} bytes)")

    # Test downloading .onnx
    onnx_resp = client.get(f"/api/train/{session.project_id}/artifact?format=onnx")
    assert onnx_resp.status_code == 200
    assert len(onnx_resp.content) > 0
    print(f"[OK] Downloaded .onnx artifact ({len(onnx_resp.content)} bytes)")


if __name__ == "__main__":
    print("Running Phase 3 Test Suite...")
    test_celery_task_direct_execution()
    test_api_async_dispatch_and_status()
    test_websocket_telemetry_stream()
    test_async_artifact_download()
    print("\n[ALL TESTS PASSED] Celery async tasks & WebSocket telemetry stream validated successfully!")
