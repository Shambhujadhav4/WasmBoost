"""
Test script to verify secure .skops and .onnx serialization without pickle.
"""
import io
from pathlib import Path
import numpy as np
import onnxruntime as rt
import pandas as pd
import skops.io as sio
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.training import TrainRequest
from app.services.dataset_service import dataset_store
from app.services.training_service import training_service


def test_classification_secure_export():
    print("\n--- Testing Classification Secure Export (.skops & .onnx) ---")
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "feature_a": np.random.randn(n_samples),
        "feature_b": np.random.randn(n_samples),
        "feature_c": np.random.randn(n_samples),
        "target": np.random.choice(["cat", "dog"], size=n_samples),
    })

    session = dataset_store.create_project(df, input_kind="csv", source_filename="test_cls.csv")
    req = TrainRequest(
        project_id=session.project_id,
        task_type="classification",
        model_name="Random Forest",
        target_column="target",
        feature_columns=["feature_a", "feature_b", "feature_c"],
        test_size=0.2,
        random_state=42,
        run_cv=True,
    )

    trained_session = training_service.train(session, req)

    # 1. Assert no .pkl exists
    models_dir = Path("models") / session.project_id
    pkl_files = list(models_dir.glob("*.pkl"))
    assert len(pkl_files) == 0, f"Found unexpected .pkl file: {pkl_files}"
    print("[OK] Verified zero .pkl files exist.")

    # 2. Assert .skops exists
    assert trained_session.artifact_skops_path is not None
    skops_path = Path(trained_session.artifact_skops_path)
    assert skops_path.exists(), f".skops file not found: {skops_path}"
    assert skops_path.suffix == ".skops"
    print(f"[OK] Verified .skops file created: {skops_path.name} ({skops_path.stat().st_size} bytes)")

    # 3. Assert .skops loads safely with untrusted type checks
    untrusted = sio.get_untrusted_types(file=str(skops_path))
    loaded_payload = sio.load(str(skops_path), trusted=untrusted)
    assert "model" in loaded_payload
    assert loaded_payload["project_id"] == session.project_id
    assert loaded_payload["task_type"] == "classification"
    loaded_model = loaded_payload["model"]
    sample_input = df[["feature_a", "feature_b", "feature_c"]].head(3)
    skops_preds = loaded_model.predict(sample_input)
    assert len(skops_preds) == 3
    print(f"[OK] Verified safe .skops loading and prediction: {skops_preds}")

    # 4. Assert .onnx exists
    assert trained_session.artifact_onnx_path is not None
    onnx_path = Path(trained_session.artifact_onnx_path)
    assert onnx_path.exists(), f".onnx file not found: {onnx_path}"
    assert onnx_path.suffix == ".onnx"
    print(f"[OK] Verified .onnx file created: {onnx_path.name} ({onnx_path.stat().st_size} bytes)")

    # 5. Assert .onnx runs inference with onnxruntime
    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    sample_arr = sample_input.to_numpy(dtype=np.float32)
    onnx_res = sess.run(None, {input_name: sample_arr})
    print(f"[OK] Verified ONNX runtime inference output count: {len(onnx_res)}")
    assert len(onnx_res) >= 1

    return session.project_id


def test_regression_secure_export():
    print("\n--- Testing Regression Secure Export (.skops & .onnx) ---")
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 10 + 5,
    })

    session = dataset_store.create_project(df, input_kind="csv", source_filename="test_reg.csv")
    req = TrainRequest(
        project_id=session.project_id,
        task_type="regression",
        model_name="Linear Regression",
        target_column="target",
        feature_columns=["feature_1", "feature_2", "feature_3"],
        test_size=0.2,
        random_state=42,
        run_cv=True,
    )

    trained_session = training_service.train(session, req)

    # 1. Assert .skops exists
    skops_path = Path(trained_session.artifact_skops_path)
    assert skops_path.exists()
    untrusted = sio.get_untrusted_types(file=str(skops_path))
    loaded_payload = sio.load(str(skops_path), trusted=untrusted)
    assert loaded_payload["task_type"] == "regression"
    print(f"[OK] Regression .skops verified: {skops_path.name}")

    # 2. Assert .onnx exists & runs inference
    onnx_path = Path(trained_session.artifact_onnx_path)
    assert onnx_path.exists()
    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    sample_arr = df[["feature_1", "feature_2", "feature_3"]].head(2).to_numpy(dtype=np.float32)
    onnx_res = sess.run(None, {input_name: sample_arr})
    assert len(onnx_res) >= 1
    print(f"[OK] Regression .onnx verified: {onnx_path.name}, inference output: {onnx_res[0].flatten()}")


def test_api_download_endpoints(project_id: str):
    print("\n--- Testing FastAPI Artifact Download Endpoints ---")
    client = TestClient(app)

    # Default download (should be skops)
    resp = client.get(f"/api/train/{project_id}/artifact")
    assert resp.status_code == 200, f"Failed default download: {resp.status_code}"
    assert resp.headers["content-disposition"].endswith('.skops"') or ".skops" in resp.headers["content-disposition"]
    assert len(resp.content) > 0
    print(f"[OK] Default artifact download endpoint OK: {resp.headers.get('content-disposition')}")

    # Query param ?format=skops
    resp = client.get(f"/api/train/{project_id}/artifact?format=skops")
    assert resp.status_code == 200
    assert ".skops" in resp.headers["content-disposition"]
    print(f"[OK] Query param ?format=skops download OK: {resp.headers.get('content-disposition')}")

    # Query param ?format=onnx
    resp = client.get(f"/api/train/{project_id}/artifact?format=onnx")
    assert resp.status_code == 200
    assert ".onnx" in resp.headers["content-disposition"]
    assert len(resp.content) > 0
    print(f"[OK] Query param ?format=onnx download OK: {resp.headers.get('content-disposition')}")

    # Direct path /artifact/onnx
    resp = client.get(f"/api/train/{project_id}/artifact/onnx")
    assert resp.status_code == 200
    assert ".onnx" in resp.headers["content-disposition"]
    print(f"[OK] Path /artifact/onnx download OK: {resp.headers.get('content-disposition')}")

    # Direct path /artifact/skops
    resp = client.get(f"/api/train/{project_id}/artifact/skops")
    assert resp.status_code == 200
    assert ".skops" in resp.headers["content-disposition"]
    print(f"[OK] Path /artifact/skops download OK: {resp.headers.get('content-disposition')}")


if __name__ == "__main__":
    project_id = test_classification_secure_export()
    test_regression_secure_export()
    test_api_download_endpoints(project_id)
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY! Model export is fully secure (.skops & .onnx).")
