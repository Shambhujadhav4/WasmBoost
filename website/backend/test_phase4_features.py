"""
Phase 4 End-to-End Test Suite for WasmBoost:
- Mutual Information ranking
- Recursive Feature Elimination (RFE)
- Optuna Bayesian Hyperparameter Optimization
- TreeSHAP Model Explainability Engine
- REST API Routes Verification
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes

from modules.preprocessing import (
    compute_mutual_information,
    recursive_feature_elimination,
    DataPreprocessor,
)
from modules.models import ModelTrainer
from modules.visualizations import DataVisualizer
from fastapi.testclient import TestClient
from app.main import app
from app.services.dataset_service import dataset_store, ProjectSession


def test_mutual_information():
    print("=== Testing Mutual Information ===")
    # 1. Classification (Iris)
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    mi_clf = compute_mutual_information(
        df_iris,
        target_col="target",
        feature_cols=list(iris.feature_names),
        task_type="classification",
    )
    assert len(mi_clf) == 4, f"Expected 4 features, got {len(mi_clf)}"
    assert all("feature" in item and "score" in item and "normalized_score" in item for item in mi_clf)
    assert mi_clf[0]["score"] >= mi_clf[-1]["score"], "MI scores should be sorted descending"
    print("[OK] Classification MI ranking passed:", [f"{x['feature']}: {x['score']:.3f}" for x in mi_clf])

    # 2. Regression (Diabetes)
    diab = load_diabetes(as_frame=True)
    df_diab = diab.frame
    mi_reg = compute_mutual_information(
        df_diab,
        target_col="target",
        feature_cols=list(diab.feature_names),
        task_type="regression",
    )
    assert len(mi_reg) == 10
    print("[OK] Regression MI ranking passed:", [f"{x['feature']}: {x['score']:.3f}" for x in mi_reg[:3]])


def test_recursive_feature_elimination():
    print("\n=== Testing Recursive Feature Elimination (RFE) ===")
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    feature_cols = list(iris.feature_names)

    # Prune 4 features down to 2
    selected, eliminated, rankings, df_pruned = recursive_feature_elimination(
        df_iris,
        target_col="target",
        feature_cols=feature_cols,
        task_type="classification",
        n_features_to_select=2,
        estimator_name="Random Forest",
    )
    assert len(selected) == 2, f"Expected 2 selected features, got {len(selected)}"
    assert len(eliminated) == 2, f"Expected 2 eliminated features, got {len(eliminated)}"
    assert len(rankings) == 4
    assert df_pruned.shape[1] == 3, f"Expected 2 features + 1 target = 3 columns, got {df_pruned.shape[1]}"
    print("[OK] RFE Feature Pruning passed. Selected features:", selected, "Eliminated:", eliminated)


def test_optuna_tuning_and_shap():
    print("\n=== Testing Optuna Tuning and TreeSHAP Explainability ===")
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    feature_cols = list(iris.feature_names)

    trainer = ModelTrainer()
    trainer.prepare_data(
        df_iris,
        target_col="target",
        feature_cols=feature_cols,
        test_size=0.2,
        random_state=42,
        task_type="classification",
    )

    trials_logged = []
    def on_trial(trial_rec):
        trials_logged.append(trial_rec)

    # 1. Run Optuna tuning
    model = trainer.tune_and_train(
        task_type="classification",
        model_name="Random Forest",
        n_trials=8,
        random_state=42,
        pruning_enabled=True,
        trial_callback=on_trial,
    )
    assert model is not None
    assert trainer.optuna_results is not None
    assert trainer.optuna_results["n_trials"] == 8
    assert len(trials_logged) == 8
    assert "best_params" in trainer.optuna_results
    assert "n_estimators" in trainer.optuna_results["best_params"]
    print(f"[OK] Optuna study completed successfully! Best score: {trainer.optuna_results['best_value']}, Best params: {trainer.optuna_results['best_params']}")

    # 2. Compute TreeSHAP explanations
    shap_results = trainer.compute_shap_explanations(sample_limit=50)
    assert shap_results is not None
    assert "feature_importance" in shap_results
    assert len(shap_results["feature_importance"]) == 4
    assert "sample_explanations" in shap_results
    assert len(shap_results["sample_explanations"]) > 0
    assert "beeswarm_points" in shap_results
    assert len(shap_results["beeswarm_points"]) > 0
    print(f"[OK] TreeSHAP computed successfully! Top feature: {shap_results['feature_importance'][0]['feature']} (importance: {shap_results['feature_importance'][0]['importance']})")

    # 3. Test Visualizers
    opt_hist_fig = DataVisualizer.plot_optuna_history(trainer.optuna_results["trials_history"])
    assert opt_hist_fig is not None
    shap_sum_fig = DataVisualizer.plot_shap_summary(shap_results)
    assert shap_sum_fig is not None
    shap_water_fig = DataVisualizer.plot_shap_waterfall(shap_results["sample_explanations"][0])
    assert shap_water_fig is not None
    shap_dep_fig = DataVisualizer.plot_shap_dependence(shap_results, feature_cols[0])
    assert shap_dep_fig is not None
    print("[OK] Plotly visualizers generated valid figures for Optuna & TreeSHAP.")


def test_api_routes():
    print("\n=== Testing API Endpoints ===")
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, res.text

    # 2. Setup project session with Iris
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    project_id = "test-phase4-proj"
    session = ProjectSession(
        project_id=project_id,
        raw_data=df_iris.copy(),
        processed_data=df_iris.copy(),
        target_column="target",
        feature_columns=list(iris.feature_names),
        task_type="classification",
    )
    dataset_store._projects[project_id] = session

    # 3. Test Mutual Information endpoint
    res_mi = client.post(
        "/api/preprocess/mutual-information",
        json={
            "project_id": project_id,
            "target_column": "target",
        },
    )
    assert res_mi.status_code == 200, res_mi.text
    mi_data = res_mi.json()
    assert len(mi_data["scores"]) == 4
    print("[OK] POST /api/preprocess/mutual-information OK")

    # 4. Test Feature Selection endpoint
    res_fs = client.post(
        "/api/preprocess/feature-selection",
        json={
            "project_id": project_id,
            "method": "mi",
            "target_column": "target",
            "n_features_to_select": 2,
        },
    )
    assert res_fs.status_code == 200, res_fs.text
    fs_data = res_fs.json()
    assert len(fs_data["selected_features"]) == 2
    print("[OK] POST /api/preprocess/feature-selection OK")

    # Reset processed data and train model with Optuna & SHAP
    session.processed_data = df_iris.copy()
    trainer = ModelTrainer()
    trainer.prepare_data(
        df_iris,
        target_col="target",
        feature_cols=list(iris.feature_names),
        task_type="classification",
    )
    trainer.tune_and_train("classification", "Random Forest", n_trials=5)
    trainer.compute_shap_explanations()
    session.model_trainer = trainer
    session.model_results = {
        "accuracy": 0.9667,
        "optuna_optimization": trainer.optuna_results,
        "shap_explanations": trainer.shap_results,
    }
    dataset_store.save_project(session)

    # 5. Test SHAP endpoint
    res_shap = client.get(f"/api/train/{project_id}/shap")
    assert res_shap.status_code == 200, res_shap.text
    assert res_shap.json()["shap_explanations"] is not None
    print("[OK] GET /api/train/{project_id}/shap OK")

    # 6. Test Visualize Optuna History endpoint
    res_opt_viz = client.get(f"/api/visualize/{project_id}/optuna-history")
    assert res_opt_viz.status_code == 200, res_opt_viz.text
    assert res_opt_viz.json()["history_figure"] is not None
    print("[OK] GET /api/visualize/{project_id}/optuna-history OK")

    # 7. Test Visualize SHAP Summary endpoint
    res_shap_sum = client.get(f"/api/visualize/{project_id}/shap-summary")
    assert res_shap_sum.status_code == 200, res_shap_sum.text
    assert res_shap_sum.json()["figure"] is not None
    print("[OK] GET /api/visualize/{project_id}/shap-summary OK")

    # 8. Test Visualize SHAP Waterfall endpoint
    res_shap_wat = client.get(f"/api/visualize/{project_id}/shap-waterfall?sample_index=0")
    assert res_shap_wat.status_code == 200, res_shap_wat.text
    assert res_shap_wat.json()["figure"] is not None
    print("[OK] GET /api/visualize/{project_id}/shap-waterfall OK")

    # 9. Test Visualize SHAP Dependence endpoint
    res_shap_dep = client.get(f"/api/visualize/{project_id}/shap-dependence?feature={iris.feature_names[0]}")
    assert res_shap_dep.status_code == 200, res_shap_dep.text
    assert res_shap_dep.json()["figure"] is not None
    print("[OK] GET /api/visualize/{project_id}/shap-dependence OK")


if __name__ == "__main__":
    test_mutual_information()
    test_recursive_feature_elimination()
    test_optuna_tuning_and_shap()
    test_api_routes()
    print("\n=== ALL PHASE 4 TESTS PASSED FLAWLESSLY! ===")
