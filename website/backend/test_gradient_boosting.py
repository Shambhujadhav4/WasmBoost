"""
Test suite validating XGBoost, LightGBM, and CatBoost integration in DataPilot:
- Model Recommendation Service & Heuristic Advice (LightGBM on large datasets, CatBoost on categorical, XGBoost on tabular)
- Model Training for Classification & Regression (XGBoost, LightGBM, CatBoost)
- Metric calculation, Cross-Validation, and Feature Importance extraction
- Secure .skops model serialization and loading
- ONNX conversion and runtime inference with onnxruntime
- FastAPI Training & Artifact Download Endpoints
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
from app.services.recommendation_service import recommendation_service
from app.services.training_service import training_service


def test_recommendation_heuristics_and_benchmarks():
    print("\n=== 1. Testing Recommendation Service Benchmarks & Insights ===")
    
    # 1A. Standard dataset recommendation
    np.random.seed(42)
    n = 100
    df_std = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
        "target": np.random.choice(["alpha", "beta"], size=n),
    })
    session_std = dataset_store.create_project(df_std, input_kind="csv", source_filename="std.csv")
    rec_std = recommendation_service.recommend(session_std)
    
    assert rec_std.recommended_task_type == "classification"
    assert rec_std.best_model is not None
    candidate_names = [c.model_name for c in rec_std.candidate_models]
    print(f"[OK] Standard benchmark evaluated candidates: {candidate_names}")
    print(f"[OK] Best model: {rec_std.best_model.model_name} (Score: {rec_std.best_model.mean_score:.4f})")
    
    # Verify XGBoost, LightGBM, CatBoost are in candidates or notes
    all_notes = " ".join(rec_std.notes)
    assert "XGBoost" in all_notes or "XGBoost" in candidate_names
    
    # 1B. Large dataset heuristic (LightGBM)
    n_large = 10500
    df_large = pd.DataFrame({
        f"feat_{i}": np.random.randn(n_large) for i in range(5)
    })
    df_large["target"] = np.random.randn(n_large)
    session_large = dataset_store.create_project(df_large, input_kind="csv", source_filename="large.csv")
    rec_large = recommendation_service.recommend(session_large)
    notes_large = " ".join(rec_large.notes)
    assert "LightGBM" in notes_large and "Large dataset" in notes_large
    print(f"[OK] Verified Large Dataset recommendation note highlights LightGBM.")

    # 1C. High-categorical heuristic (CatBoost)
    n_cat = 200
    df_cat = pd.DataFrame({
        "cat_1": np.random.choice(["red", "green", "blue"], size=n_cat),
        "cat_2": np.random.choice(["small", "medium", "large"], size=n_cat),
        "cat_3": np.random.choice(["tier1", "tier2", "tier3"], size=n_cat),
        "cat_4": np.random.choice(["north", "south", "east", "west"], size=n_cat),
        "num_1": np.random.randn(n_cat),
        "target": np.random.choice(["approved", "rejected"], size=n_cat),
    })
    session_cat = dataset_store.create_project(df_cat, input_kind="csv", source_filename="cat.csv")
    rec_cat = recommendation_service.recommend(session_cat)
    notes_cat = " ".join(rec_cat.notes)
    assert "CatBoost" in notes_cat and "categorical" in notes_cat.lower()
    print(f"[OK] Verified High-Categorical dataset recommendation note highlights CatBoost.")


def test_gradient_boosting_classification():
    print("\n=== 2. Testing Gradient Boosting Models (Classification) ===")
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "num_a": np.random.randn(n),
        "num_b": np.random.randn(n) * 2 + 1,
        "num_c": np.random.randn(n) - 0.5,
        "target": np.random.choice(["class_A", "class_B"], size=n),
    })

    models_to_test = ["XGBoost", "LightGBM", "CatBoost"]
    for model_name in models_to_test:
        print(f"\n--- Testing Classification: {model_name} ---")
        session = dataset_store.create_project(df, input_kind="csv", source_filename=f"test_{model_name}_cls.csv")
        req = TrainRequest(
            project_id=session.project_id,
            task_type="classification",
            model_name=model_name,
            target_column="target",
            feature_columns=["num_a", "num_b", "num_c"],
            test_size=0.2,
            random_state=42,
            run_cv=True,
        )

        trained_session = training_service.train(session, req)

        # 1. Check Metrics
        results = trained_session.model_results
        assert results is not None
        assert "accuracy" in results
        assert "f1_score" in results
        assert "cv_scores" in results
        assert len(results["cv_scores"]) == 5
        print(f"[OK] {model_name} Accuracy: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}, CV Mean: {np.mean(results['cv_scores']):.4f}")

        # 2. Check Feature Importance
        imp_df = session.model_trainer.get_feature_importance()
        assert imp_df is not None
        assert len(imp_df) == 3
        print(f"[OK] {model_name} Feature Importances:\n{imp_df.to_string(index=False)}")

        # 3. Check .skops Secure Export & Load
        skops_path = Path(trained_session.artifact_skops_path)
        assert skops_path.exists()
        untrusted = sio.get_untrusted_types(file=str(skops_path))
        loaded_payload = sio.load(str(skops_path), trusted=untrusted)
        assert loaded_payload["model_name"] == model_name
        loaded_model = loaded_payload["model"]
        sample_input = df[["num_a", "num_b", "num_c"]].head(3)
        skops_preds = loaded_model.predict(sample_input)
        assert len(skops_preds) == 3
        print(f"[OK] {model_name} .skops exported & verified ({skops_path.stat().st_size} bytes)")

        # 4. Check .onnx Export & Inference
        onnx_path = Path(trained_session.artifact_onnx_path)
        assert onnx_path.exists()
        sess = rt.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        sample_arr = sample_input.to_numpy(dtype=np.float32)
        onnx_res = sess.run(None, {input_name: sample_arr})
        assert len(onnx_res) >= 1
        print(f"[OK] {model_name} .onnx exported & ONNX Runtime inference verified ({onnx_path.stat().st_size} bytes)")


def test_gradient_boosting_regression():
    print("\n=== 3. Testing Gradient Boosting Models (Regression) ===")
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n) * 3,
        "x3": np.random.randn(n) + 2,
        "target": np.random.randn(n) * 10 + 25,
    })

    models_to_test = ["XGBoost", "LightGBM", "CatBoost"]
    for model_name in models_to_test:
        print(f"\n--- Testing Regression: {model_name} ---")
        session = dataset_store.create_project(df, input_kind="csv", source_filename=f"test_{model_name}_reg.csv")
        req = TrainRequest(
            project_id=session.project_id,
            task_type="regression",
            model_name=model_name,
            target_column="target",
            feature_columns=["x1", "x2", "x3"],
            test_size=0.2,
            random_state=42,
            run_cv=True,
        )

        trained_session = training_service.train(session, req)

        # 1. Check Metrics
        results = trained_session.model_results
        assert results is not None
        assert "r2_score" in results
        assert "rmse" in results
        assert "cv_scores" in results
        print(f"[OK] {model_name} R2: {results['r2_score']:.4f}, RMSE: {results['rmse']:.4f}")

        # 2. Check Feature Importance
        imp_df = session.model_trainer.get_feature_importance()
        assert imp_df is not None
        assert len(imp_df) == 3

        # 3. Check .skops Export
        skops_path = Path(trained_session.artifact_skops_path)
        assert skops_path.exists()
        untrusted = sio.get_untrusted_types(file=str(skops_path))
        loaded_payload = sio.load(str(skops_path), trusted=untrusted)
        assert loaded_payload["task_type"] == "regression"
        print(f"[OK] {model_name} regression .skops verified.")

        # 4. Check .onnx Export & Inference
        onnx_path = Path(trained_session.artifact_onnx_path)
        assert onnx_path.exists()
        sess = rt.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        sample_arr = df[["x1", "x2", "x3"]].head(2).to_numpy(dtype=np.float32)
        onnx_res = sess.run(None, {input_name: sample_arr})
        assert len(onnx_res) >= 1
        print(f"[OK] {model_name} regression .onnx verified. Sample prediction: {onnx_res[0].flatten()}")


def test_api_endpoints_gradient_boosting():
    print("\n=== 4. Testing FastAPI Endpoints for Gradient Boosting Models ===")
    client = TestClient(app)

    # 1. Upload CSV
    np.random.seed(42)
    n = 100
    csv_content = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "label": np.random.choice([0, 1], size=n),
    }).to_csv(index=False).encode("utf-8")

    upload_resp = client.post(
        "/api/upload/file",
        files={"file": ("dataset.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert upload_resp.status_code == 200
    project_id = upload_resp.json()["project_id"]
    print(f"[OK] Uploaded dataset to project {project_id}")

    # 2. Get recommendations
    rec_resp = client.get(f"/api/upload/{project_id}/workflow-recommendation")
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["best_model"] is not None
    print(f"[OK] Recommendation returned best model: {rec_data['best_model']['model_name']}")

    # 3. Train XGBoost (Async Celery dispatch)
    from app.core.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    celery_app.conf.result_backend = "cache+memory://"
    celery_app.conf.task_store_eager_result = True


    train_resp = client.post(
        "/api/train",
        json={
            "project_id": project_id,
            "task_type": "classification",
            "model_name": "XGBoost",
            "target_column": "label",
            "feature_columns": ["f1", "f2"],
            "test_size": 0.2,
            "random_state": 42,
            "run_cv": True,
        },
    )
    assert train_resp.status_code == 202
    train_data = train_resp.json()
    assert "task_id" in train_data
    task_id = train_data["task_id"]

    # Verify session has artifacts
    updated_session = dataset_store.get_project(project_id)
    assert updated_session.artifact_skops_path is not None
    assert updated_session.artifact_onnx_path is not None
    print(f"[OK] API Training for XGBoost dispatched task {task_id} and generated artifacts: skops={Path(updated_session.artifact_skops_path).name}, onnx={Path(updated_session.artifact_onnx_path).name}")


    # 4. Download .skops
    skops_resp = client.get(f"/api/train/{project_id}/artifact?format=skops")
    assert skops_resp.status_code == 200
    assert ".skops" in skops_resp.headers.get("content-disposition", "")
    print(f"[OK] Downloaded .skops artifact ({len(skops_resp.content)} bytes)")

    # 5. Download .onnx
    onnx_resp = client.get(f"/api/train/{project_id}/artifact?format=onnx")
    assert onnx_resp.status_code == 200
    assert ".onnx" in onnx_resp.headers.get("content-disposition", "")
    print(f"[OK] Downloaded .onnx artifact ({len(onnx_resp.content)} bytes)")


if __name__ == "__main__":
    test_recommendation_heuristics_and_benchmarks()
    test_gradient_boosting_classification()
    test_gradient_boosting_regression()
    test_api_endpoints_gradient_boosting()
    print("\n[ALL TESTS PASSED SUCCESSFULLY] XGBoost, LightGBM, and CatBoost are fully integrated!")
