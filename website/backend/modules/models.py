import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class ModelTrainer:
    """Trains and evaluates classification and regression models."""

    CLASSIFICATION_MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(iterations=100, random_state=42, verbose=0),
    }

    REGRESSION_MODELS = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Lasso Regression": Lasso(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostRegressor(iterations=100, random_state=42, verbose=0),
    }

    def __init__(self):
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.feature_names: list = []
        self.task_type: str = ""
        self.model_name: str = ""
        self.label_encoder: LabelEncoder | None = None

        self.optuna_results: dict | None = None
        self.shap_results: dict | None = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        test_size: float = 0.2,
        random_state: int = 42,
        task_type: str | None = None,
    ):
        X = df[feature_cols]
        y = df[target_col]
        self.feature_names = feature_cols
        self.task_type = task_type or ""

        stratify = None
        if task_type == "classification" and y.nunique(dropna=False) > 1:
            class_counts = y.value_counts(dropna=False)
            if class_counts.min() < 2:
                raise ValueError(
                    "Each target class needs at least 2 rows for stratified splitting."
                )

            test_count = (
                int(np.ceil(len(y) * test_size))
                if isinstance(test_size, float)
                else int(test_size)
            )
            train_count = len(y) - test_count
            n_classes = y.nunique(dropna=False)
            if test_count < n_classes or train_count < n_classes:
                raise ValueError(
                    "Adjust the test split so both train and test sets can include every target class."
                )

            stratify = y

        # For classification, encode string/object classes so gradient boosting models work seamlessly
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            self.label_encoder = LabelEncoder()
            y_to_split = pd.Series(self.label_encoder.fit_transform(y), index=y.index, name=y.name)
            if stratify is not None:
                stratify = y_to_split
        else:
            self.label_encoder = None
            y_to_split = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y_to_split,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, task_type: str, model_name: str):
        self.task_type = task_type
        self.model_name = model_name

        registry = (
            self.CLASSIFICATION_MODELS
            if task_type == "classification"
            else self.REGRESSION_MODELS
        )

        if model_name not in registry:
            raise ValueError(f"Model '{model_name}' not found in registry.")

        import sklearn.base as base

        self.model = base.clone(registry[model_name])
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        if task_type == "classification" and hasattr(self.model, "predict_proba"):
            self.y_pred_proba = self.model.predict_proba(self.X_test)

        return self.model

    def _build_model_with_params(self, task_type: str, model_name: str, params: dict):
        if task_type == "classification":
            if model_name == "Random Forest":
                return RandomForestClassifier(random_state=42, **params)
            if model_name == "Gradient Boosting":
                return GradientBoostingClassifier(random_state=42, **params)
            if model_name == "XGBoost":
                return xgb.XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0, **params)
            if model_name == "LightGBM":
                return lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
            if model_name == "CatBoost":
                return cb.CatBoostClassifier(random_state=42, verbose=0, **params)
            if model_name == "Logistic Regression":
                return LogisticRegression(max_iter=1000, random_state=42, **params)
            if model_name == "Decision Tree":
                return DecisionTreeClassifier(random_state=42, **params)
            if model_name == "SVM":
                return SVC(probability=True, random_state=42, **params)
            if model_name == "K-Nearest Neighbors":
                return KNeighborsClassifier(**params)
            return RandomForestClassifier(random_state=42, **params)
        else:
            if model_name == "Random Forest":
                return RandomForestRegressor(random_state=42, **params)
            if model_name == "Gradient Boosting":
                return GradientBoostingRegressor(random_state=42, **params)
            if model_name == "XGBoost":
                return xgb.XGBRegressor(random_state=42, verbosity=0, **params)
            if model_name == "LightGBM":
                return lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
            if model_name == "CatBoost":
                return cb.CatBoostRegressor(random_state=42, verbose=0, **params)
            if model_name == "Linear Regression":
                return LinearRegression(**params)
            if model_name == "Ridge Regression":
                return Ridge(random_state=42, **params)
            if model_name == "Lasso Regression":
                return Lasso(random_state=42, **params)
            if model_name == "Decision Tree":
                return DecisionTreeRegressor(random_state=42, **params)
            if model_name == "SVR":
                return SVR(**params)
            if model_name == "K-Nearest Neighbors":
                return KNeighborsRegressor(**params)
            return RandomForestRegressor(random_state=42, **params)

    def tune_and_train(
        self,
        task_type: str,
        model_name: str,
        n_trials: int = 15,
        random_state: int = 42,
        pruning_enabled: bool = True,
        tuning_metric: str | None = None,
        trial_callback: object | None = None,
    ):
        import time
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.task_type = task_type
        self.model_name = model_name

        scoring = tuning_metric or ("accuracy" if task_type == "classification" else "r2")
        cv_folds = 3 if len(self.X_train) >= 30 else 2

        def suggest_params(trial: optuna.Trial) -> dict:
            p = {}
            if "Random Forest" in model_name:
                p["n_estimators"] = trial.suggest_int("n_estimators", 30, 150, step=10)
                p["max_depth"] = trial.suggest_int("max_depth", 3, 16)
                p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
                p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 4)
            elif "Gradient Boosting" in model_name:
                p["n_estimators"] = trial.suggest_int("n_estimators", 30, 120, step=10)
                p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
                p["max_depth"] = trial.suggest_int("max_depth", 3, 8)
                p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)
            elif "XGBoost" in model_name:
                p["n_estimators"] = trial.suggest_int("n_estimators", 30, 120, step=10)
                p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
                p["max_depth"] = trial.suggest_int("max_depth", 3, 8)
                p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)
                p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1)
            elif "LightGBM" in model_name:
                p["n_estimators"] = trial.suggest_int("n_estimators", 30, 120, step=10)
                p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
                p["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
                p["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            elif "CatBoost" in model_name:
                p["iterations"] = trial.suggest_int("iterations", 30, 120, step=10)
                p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
                p["depth"] = trial.suggest_int("depth", 4, 8)
                p["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
            elif "Logistic" in model_name or "Ridge" in model_name or "Lasso" in model_name:
                if "Logistic" in model_name:
                    p["C"] = trial.suggest_float("C", 1e-3, 50.0, log=True)
                else:
                    p["alpha"] = trial.suggest_float("alpha", 1e-3, 50.0, log=True)
            elif "SVM" in model_name or "SVR" in model_name:
                p["C"] = trial.suggest_float("C", 0.1, 30.0, log=True)
                p["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            elif "Neighbors" in model_name:
                p["n_neighbors"] = trial.suggest_int("n_neighbors", 3, min(20, max(3, len(self.X_train) - 1)))
                p["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            elif "Decision Tree" in model_name:
                p["max_depth"] = trial.suggest_int("max_depth", 3, 16)
                p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
            return p

        study_history: list[dict] = []
        best_so_far = float("-inf")

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_so_far
            t_start = time.time()
            params = suggest_params(trial)
            est = self._build_model_with_params(task_type, model_name, params)
            scores = cross_val_score(est, self.X_train, self.y_train, cv=cv_folds, scoring=scoring)
            mean_score = float(scores.mean())
            if np.isnan(mean_score):
                mean_score = float("-inf")

            if mean_score > best_so_far:
                best_so_far = mean_score

            duration = round(time.time() - t_start, 3)
            trial_record = {
                "trial_number": trial.number + 1,
                "value": round(mean_score, 4) if mean_score != float("-inf") else None,
                "best_value": round(best_so_far, 4) if best_so_far != float("-inf") else None,
                "params": params,
                "state": "COMPLETE",
                "duration_seconds": duration,
            }
            study_history.append(trial_record)

            if callable(trial_callback):
                trial_callback(trial_record)

            return mean_score

        sampler = optuna.samplers.TPESampler(seed=random_state)
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
            if pruning_enabled
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params = study.best_params
        best_score = float(study.best_value)

        # Calculate parameter importances if multiple completed trials
        param_importances: dict[str, float] = {}
        try:
            if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 3:
                raw_importances = optuna.importance.get_param_importances(study)
                param_importances = {k: round(float(v), 4) for k, v in raw_importances.items()}
        except Exception:
            pass

        self.optuna_results = {
            "study_name": study.study_name,
            "best_params": best_params,
            "best_value": round(best_score, 4),
            "direction": "maximize",
            "metric_name": scoring,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "trials_history": study_history,
            "param_importances": param_importances,
        }

        # Train final model on full X_train with best_params
        self.model = self._build_model_with_params(task_type, model_name, best_params)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        if task_type == "classification" and hasattr(self.model, "predict_proba"):
            self.y_pred_proba = self.model.predict_proba(self.X_test)

        return self.model

    def compute_shap_explanations(self, sample_limit: int = 200) -> dict | None:
        """
        Computes TreeSHAP feature explanations and local contribution breakdowns.
        Supports tree-based algorithms and provides a clean fallback for linear models.
        """
        if self.model is None or self.X_test is None or len(self.X_test) == 0:
            return None

        import shap

        model_type_name = self.model.__class__.__name__
        is_tree_model = any(
            t in model_type_name
            for t in ["Forest", "GradientBoosting", "XGB", "LGBM", "CatBoost", "DecisionTree"]
        )

        n_samples = min(sample_limit, len(self.X_test))
        X_sample = self.X_test.iloc[:n_samples].copy()

        try:
            if is_tree_model:
                explainer = shap.TreeExplainer(self.model)
                shap_raw = explainer.shap_values(X_sample)
                expected_val = getattr(explainer, "expected_value", None)
            elif "Logistic" in model_type_name or "Ridge" in model_type_name or "Lasso" in model_type_name or "Linear" in model_type_name:
                bg = self.X_train.sample(min(100, len(self.X_train)), random_state=42)
                explainer = shap.LinearExplainer(self.model, bg)
                shap_raw = explainer.shap_values(X_sample)
                expected_val = getattr(explainer, "expected_value", None)
            else:
                bg = self.X_train.sample(min(40, len(self.X_train)), random_state=42)
                explainer = shap.Explainer(self.model.predict, bg)
                shap_exp = explainer(X_sample)
                shap_raw = shap_exp.values
                expected_val = getattr(shap_exp, "base_values", None)

            # Normalize SHAP values into 2D numpy array [n_samples, n_features]
            if isinstance(shap_raw, list):
                # Multi-class classification: pick positive class (binary) or average across classes
                if len(shap_raw) == 2:
                    shap_mat = np.array(shap_raw[1])
                    base_val = float(expected_val[1]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val or 0.0)
                else:
                    shap_mat = np.mean(np.array([np.abs(c) for c in shap_raw]), axis=0)
                    base_val = float(np.mean(expected_val)) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val or 0.0)
            elif isinstance(shap_raw, np.ndarray):
                if shap_raw.ndim == 3:
                    # [n_samples, n_features, n_classes]
                    if shap_raw.shape[2] == 2:
                        shap_mat = shap_raw[:, :, 1]
                        base_val = float(expected_val[1]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val or 0.0)
                    else:
                        shap_mat = np.mean(np.abs(shap_raw), axis=2)
                        base_val = float(np.mean(expected_val)) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val or 0.0)
                else:
                    shap_mat = shap_raw
                    base_val = float(expected_val[0]) if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 0 else float(expected_val or 0.0)
            else:
                shap_mat = np.array(shap_raw)
                base_val = 0.0

            # 1. Global Mean |SHAP| values per feature
            mean_abs = np.mean(np.abs(shap_mat), axis=0)
            importance_list = [
                {"feature": str(col), "importance": round(float(mean_abs[i]), 4)}
                for i, col in enumerate(self.feature_names)
            ]
            importance_list.sort(key=lambda x: x["importance"], reverse=True)

            # 2. Beeswarm points for top 12 features across samples
            top_features = [item["feature"] for item in importance_list[:12]]
            beeswarm_points = []
            for j in range(len(X_sample)):
                for f in top_features:
                    f_idx = self.feature_names.index(f)
                    f_val = X_sample.iloc[j][f]
                    f_val_num = float(f_val) if isinstance(f_val, (int, float, np.number)) and not np.isnan(f_val) else 0.0
                    beeswarm_points.append({
                        "feature": f,
                        "feature_value": round(f_val_num, 4),
                        "shap_value": round(float(shap_mat[j, f_idx]), 4),
                        "sample_index": j,
                    })

            # 3. Local explanations for up to 20 samples
            n_local = min(20, len(X_sample))
            sample_explanations = []
            for k in range(n_local):
                contributions = []
                for i, f in enumerate(self.feature_names):
                    contributions.append({
                        "feature": f,
                        "value": round(float(X_sample.iloc[k][f]), 4) if isinstance(X_sample.iloc[k][f], (int, float, np.number)) else str(X_sample.iloc[k][f]),
                        "shap_value": round(float(shap_mat[k, i]), 4),
                    })
                contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

                pred_val = float(self.y_pred[k]) if hasattr(self.y_pred[k], "__float__") else 0.0
                sample_explanations.append({
                    "sample_index": k,
                    "base_value": round(base_val, 4),
                    "output_value": round(pred_val, 4),
                    "contributions": contributions,
                })

            self.shap_results = {
                "feature_importance": importance_list,
                "base_value": round(base_val, 4),
                "sample_explanations": sample_explanations,
                "beeswarm_points": beeswarm_points,
                "model_framework": "TreeSHAP" if is_tree_model else "SHAP Explainer",
                "is_tree_model": is_tree_model,
            }
            return self.shap_results

        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to compute SHAP explanations: %s", exc)
            return None

    def get_metrics(self) -> dict:
        return (
            self._classification_metrics()
            if self.task_type == "classification"
            else self._regression_metrics()
        )

    def _classification_metrics(self) -> dict:
        if self.label_encoder is not None:
            classes = list(range(len(self.label_encoder.classes_)))
            display_classes = list(self.label_encoder.classes_)
            y_test_eval = self.y_test
            y_pred_eval = self.y_pred
        else:
            classes = list(
                getattr(self.model, "classes_", pd.Index(self.y_test).drop_duplicates().tolist())
            )
            display_classes = classes
            y_test_eval = self.y_test
            y_pred_eval = self.y_pred

        average = "binary" if len(classes) == 2 else "macro"
        pos_label = classes[1] if len(classes) == 2 else None

        precision_kwargs = {"zero_division": 0, "average": average}
        recall_kwargs = {"zero_division": 0, "average": average}
        f1_kwargs = {"average": average}
        if pos_label is not None:
            precision_kwargs["pos_label"] = pos_label
            recall_kwargs["pos_label"] = pos_label
            f1_kwargs["pos_label"] = pos_label

        metrics = {
            "accuracy": accuracy_score(y_test_eval, y_pred_eval),
            "f1_score": f1_score(y_test_eval, y_pred_eval, **f1_kwargs),
            "precision": precision_score(y_test_eval, y_pred_eval, **precision_kwargs),
            "recall": recall_score(y_test_eval, y_pred_eval, **recall_kwargs),
            "metric_average": average,
            "positive_class": pos_label,
            "confusion_matrix": confusion_matrix(y_test_eval, y_pred_eval, labels=classes),
            "classification_report": classification_report(
                y_test_eval,
                y_pred_eval,
                labels=classes,
                target_names=[str(c) for c in display_classes] if len(display_classes) == len(classes) else None,
                zero_division=0,
            ),
        }
        if self.y_pred_proba is not None:
            try:
                if len(classes) == 2:
                    metrics["roc_auc"] = roc_auc_score(
                        self.y_test,
                        self.y_pred_proba[:, 1],
                        pos_label=pos_label,
                    )
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        self.y_test,
                        self.y_pred_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
            except Exception:
                pass
        return metrics

    def _regression_metrics(self) -> dict:
        mse = mean_squared_error(self.y_test, self.y_pred)
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(self.y_test, self.y_pred),
            "r2_score": r2_score(self.y_test, self.y_pred),
        }

    def get_feature_importance(self) -> pd.DataFrame | None:
        if hasattr(self.model, "feature_importances_"):
            return (
                pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": self.model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            return (
                pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": np.abs(coef),
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        return None

    def get_cross_val_scores(self, cv: int = 5) -> np.ndarray:
        scoring = "accuracy" if self.task_type == "classification" else "r2"
        X_all = pd.concat([self.X_train, self.X_test])
        y_all = pd.concat([self.y_train, self.y_test])
        import sklearn.base as base

        fresh_model = base.clone(self.model)
        return cross_val_score(fresh_model, X_all, y_all, cv=cv, scoring=scoring)

