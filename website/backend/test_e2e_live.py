"""
End-to-end live test script validating real HTTP requests against the live FastAPI server.
"""
import io
from pathlib import Path
import httpx
import numpy as np
import onnxruntime as rt
import pandas as pd
import skops.io as sio

BASE_URL = "http://127.0.0.1:8000/api"


def run_live_e2e_test():
    print("=== LIVE E2E VERIFICATION ===")
    
    # 1. Health check
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
        print("[OK] Backend server is healthy: 127.0.0.1:8000/api/health")

        # 2. Upload dataset
        csv_data = "age,salary,purchased\n25,50000,0\n30,60000,1\n35,80000,1\n40,90000,0\n45,110000,1\n50,120000,1\n22,45000,0\n28,52000,0\n38,85000,1\n42,95000,0\n"
        files = {"file": ("test_customers.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
        data = {"separator": ",", "encoding": "utf-8", "header_row": "0"}
        
        upload_resp = client.post("/upload/file", files=files, data=data)
        assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
        project_summary = upload_resp.json()
        project_id = project_summary["project_id"]
        print(f"[OK] Uploaded dataset successfully. Project ID: {project_id}")

        # 3. Train model
        train_payload = {
            "project_id": project_id,
            "task_type": "classification",
            "model_name": "Random Forest",
            "target_column": "purchased",
            "feature_columns": ["age", "salary"],
            "test_size": 0.2,
            "random_state": 42,
            "run_cv": True,
        }
        train_resp = client.post("/train", json=train_payload)
        assert train_resp.status_code == 200, f"Training failed: {train_resp.text}"
        snapshot = train_resp.json()
        print(f"[OK] Trained Random Forest model successfully.")
        assert snapshot["artifact_available"] is True
        assert snapshot["skops_artifact_available"] is True
        assert snapshot["onnx_artifact_available"] is True
        print(f"[OK] ProjectSnapshot confirms skops_artifact_available={snapshot['skops_artifact_available']}, onnx_artifact_available={snapshot['onnx_artifact_available']}")
        print(f"[OK] skops_filename='{snapshot['skops_artifact_filename']}', onnx_filename='{snapshot['onnx_artifact_filename']}'")

        # 4. Download .skops artifact
        skops_resp = client.get(f"/train/{project_id}/artifact?format=skops")
        assert skops_resp.status_code == 200, f"Failed .skops download: {skops_resp.status_code}"
        assert ".skops" in skops_resp.headers.get("content-disposition", "")
        skops_bytes = skops_resp.content
        assert len(skops_bytes) > 0
        print(f"[OK] Downloaded .skops binary ({len(skops_bytes)} bytes). Header: {skops_resp.headers.get('content-disposition')}")

        # Validate .skops content with skops.io
        untrusted = sio.get_untrusted_types(data=skops_bytes)
        loaded = sio.load(io.BytesIO(skops_bytes), trusted=untrusted)
        assert "model" in loaded
        sample_df = pd.DataFrame({"age": [33.0], "salary": [75000.0]})
        skops_pred = loaded["model"].predict(sample_df)
        print(f"[OK] Deserialized .skops and made prediction: {skops_pred}")

        # 5. Download .onnx artifact
        onnx_resp = client.get(f"/train/{project_id}/artifact?format=onnx")
        assert onnx_resp.status_code == 200, f"Failed .onnx download: {onnx_resp.status_code}"
        assert ".onnx" in onnx_resp.headers.get("content-disposition", "")
        onnx_bytes = onnx_resp.content
        assert len(onnx_bytes) > 0
        print(f"[OK] Downloaded .onnx binary ({len(onnx_bytes)} bytes). Header: {onnx_resp.headers.get('content-disposition')}")

        # Validate .onnx content with onnxruntime
        sess = rt.InferenceSession(onnx_bytes)
        input_name = sess.get_inputs()[0].name
        onnx_pred = sess.run(None, {input_name: np.array([[33.0, 75000.0]], dtype=np.float32)})
        print(f"[OK] Ran inference on downloaded .onnx artifact: {onnx_pred}")

        # 6. Verify zero pickle files exist in project directory
        models_dir = Path("models") / project_id
        pkl_files = list(models_dir.glob("*.pkl"))
        assert len(pkl_files) == 0, f"Found unexpected pickle files: {pkl_files}"
        print("[OK] Confirmed: No .pkl files exist in storage.")

    print("\n=== [ALL LIVE E2E CHECKS PASSED] ===")


if __name__ == "__main__":
    run_live_e2e_test()
