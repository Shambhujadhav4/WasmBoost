"""
Phase 5 End-to-End Test Suite for DataPilot:
- Client-Side WebAssembly (Pyodide) In-Browser Preprocessing Simulation
- Backend Training Handoff Engine
- On-the-Fly Session Ingestion
- Celery / Async Heavy Model Training with Optuna & TreeSHAP
- .skops & .onnx Artifact Verification
"""
from __future__ import annotations

import io
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient

from app.main import app
from app.services.dataset_service import dataset_store


def test_client_side_handoff_and_training():
    print("=== Testing Phase 5: Client-Side Pyodide Handoff & Backend Training ===")
    client = TestClient(app)

    # 1. Simulate client-side Pyodide preprocessing in WebAssembly
    iris = load_iris(as_frame=True)
    df_raw = iris.frame.copy()
    
    # Client applies StandardScaler and drops missing values in browser
    scaler = StandardScaler()
    feature_cols = list(iris.feature_names)
    df_preprocessed = df_raw.copy()
    df_preprocessed[feature_cols] = scaler.fit_transform(df_preprocessed[feature_cols])

    # Convert to client-side exported records and CSV
    preprocessed_records = df_preprocessed.to_dict(orient="records")
    preprocessed_csv = df_preprocessed.to_csv(index=False)
    
    project_id = "wasm-proj-test-001"
    print(f"[OK] Client-side preprocessing simulated: {len(preprocessed_records)} rows preprocessed in WebAssembly.")

    # 2. Transmit preprocessed payload to FastAPI training handoff
    train_payload = {
        "project_id": project_id,
        "task_type": "classification",
        "model_name": "Random Forest",
        "target_column": "target",
        "feature_columns": feature_cols,
        "test_size": 0.2,
        "random_state": 42,
        "run_cv": True,
        "use_hyperparameter_tuning": True,
        "n_trials": 6,
        "pruning_enabled": True,
        "preprocessed_data": preprocessed_records,
    }

    print("[*] Dispatching training handoff to POST /api/train...")
    res = client.post("/api/train", json=train_payload)
    assert res.status_code == 202, f"Expected 202, got {res.status_code}: {res.text}"
    task_info = res.json()
    assert task_info["status"] == "queued"
    print(f"[OK] Training task accepted: task_id={task_info['task_id']}")

    # 3. Verify session was created on-the-fly from handoff in dataset_store
    session = dataset_store.get_project(project_id)
    assert session is not None
    assert session.input_kind == "pyodide_wasm"
    assert session.processed_data.shape[0] == 150
    assert session.target_column == "target"
    print(f"[OK] On-the-fly ProjectSession verified: shape={session.processed_data.shape}, kind={session.input_kind}")

    # 4. Check training completion and results
    # Since Celery fallback executes synchronously in test environment, model_results is populated
    assert session.model_results is not None
    assert "accuracy" in session.model_results
    assert session.model_results["accuracy"] > 0.85
    assert "optuna_optimization" in session.model_results
    assert session.model_results["optuna_optimization"]["n_trials"] == 6
    assert "shap_explanations" in session.model_results
    print(f"[OK] Model training completed: Accuracy={session.model_results['accuracy']:.4f}")
    print(f"[OK] Optuna best parameters: {session.model_results['optuna_optimization']['best_params']}")
    print(f"[OK] TreeSHAP top feature: {session.model_results['shap_explanations']['feature_importance'][0]['feature']}")

    # 5. Verify Artifact generation (.skops and .onnx)
    assert session.artifact_skops_path is not None
    assert session.artifact_onnx_path is not None
    print(f"[OK] Saved .skops artifact: {session.artifact_skops_filename}")
    print(f"[OK] Saved .onnx artifact: {session.artifact_onnx_filename}")

    # 6. Verify Visualizations and SHAP endpoints with handoff session
    res_shap = client.get(f"/api/visualize/{project_id}/shap-summary")
    assert res_shap.status_code == 200, res_shap.text
    assert res_shap.json()["figure"] is not None
    print("[OK] GET /api/visualize/{project_id}/shap-summary OK")

    res_opt = client.get(f"/api/visualize/{project_id}/optuna-history")
    assert res_opt.status_code == 200, res_opt.text
    assert res_opt.json()["history_figure"] is not None
    print("[OK] GET /api/visualize/{project_id}/optuna-history OK")

    print("\n=== ALL PHASE 5 HANDOFF & TRAINING TESTS PASSED! ===")


if __name__ == "__main__":
    test_client_side_handoff_and_training()
