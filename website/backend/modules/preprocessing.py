from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """Handles all data preprocessing steps."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.label_encoders: dict = {}
        self.scaler = None
        self.scaled_columns: list = []

    def get_missing_info(self) -> pd.DataFrame:
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        return (
            pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
            .sort_values("Missing Count", ascending=False)
        )

    def drop_columns(self, columns: list) -> pd.DataFrame:
        self.df = self.df.drop(columns=columns, errors="ignore")
        return self.df

    def handle_missing_values(
        self,
        strategy: str,
        columns: list | None = None,
        fill_value=None,
    ) -> pd.DataFrame:
        cols = columns if columns else self.df.columns.tolist()

        if strategy == "drop_rows":
            self.df = self.df.dropna(subset=cols)

        elif strategy == "drop_columns":
            self.df = self.df.drop(columns=cols, errors="ignore")

        elif strategy in ("mean", "median"):
            num_cols = [c for c in cols if is_numeric_dtype(self.df[c])]
            for col in num_cols:
                val = self.df[col].mean() if strategy == "mean" else self.df[col].median()
                self.df[col] = self.df[col].fillna(val)

        elif strategy == "mode":
            for col in cols:
                mode_vals = self.df[col].mode()
                if not mode_vals.empty:
                    self.df[col] = self.df[col].fillna(mode_vals[0])

        elif strategy == "zero":
            for col in cols:
                self.df[col] = self.df[col].fillna(0)

        elif strategy == "custom" and fill_value is not None:
            for col in cols:
                self.df[col] = self.df[col].fillna(fill_value)

        return self.df

    def encode_categorical(self, method: str, columns: list | None = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not columns:
            return self.df

        if method == "label":
            for col in columns:
                non_null_mask = self.df[col].notna()
                if not non_null_mask.any():
                    self.df[col] = pd.Series(np.nan, index=self.df.index, dtype="float64")
                    continue

                le = LabelEncoder()
                encoded = pd.Series(np.nan, index=self.df.index, dtype="float64")
                encoded.loc[non_null_mask] = le.fit_transform(
                    self.df.loc[non_null_mask, col].astype(str)
                )
                self.df[col] = encoded
                self.label_encoders[col] = le

        elif method == "onehot":
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=False)

        return self.df

    def scale_features(self, method: str, columns: list | None = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            return self.df

        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        self.scaler = scalers.get(method, StandardScaler())
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        self.scaled_columns = columns
        return self.df

    def handle_outliers(
        self,
        method: str,
        columns: list | None = None,
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if method == "iqr_remove":
            mask = pd.Series([True] * len(self.df), index=self.df.index)
            for col in columns:
                q1, q3 = self.df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                mask &= ~(
                    (self.df[col] < q1 - threshold * iqr)
                    | (self.df[col] > q3 + threshold * iqr)
                )
            self.df = self.df[mask]

        elif method == "iqr_cap":
            for col in columns:
                q1, q3 = self.df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.df[col] = self.df[col].clip(
                    lower=q1 - threshold * iqr,
                    upper=q3 + threshold * iqr,
                )

        elif method == "zscore_remove":
            mask = pd.Series([True] * len(self.df), index=self.df.index)
            for col in columns:
                col_filled = self.df[col].fillna(self.df[col].mean())
                std = col_filled.std()
                if std == 0:
                    continue
                z = (col_filled - col_filled.mean()) / std
                mask &= z.abs() < threshold
            self.df = self.df[mask]

        return self.df

    def select_features_mi(
        self,
        target_col: str,
        n_features_to_select: int = 5,
        task_type: str | None = None,
        feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str], list[str], list[dict[str, Any]]]:
        ranking_info = compute_mutual_information(
            self.df,
            target_col=target_col,
            feature_cols=feature_cols,
            task_type=task_type,
        )
        selected = [item["feature"] for item in ranking_info[:n_features_to_select]]
        eliminated = [item["feature"] for item in ranking_info[n_features_to_select:]]
        # Keep selected features and any other columns (e.g. target or metadata)
        cols_to_keep = [col for col in self.df.columns if col not in eliminated]
        self.df = self.df[cols_to_keep]
        rankings = [
            {
                "feature": item["feature"],
                "ranking": idx + 1,
                "selected": item["feature"] in selected,
                "score": item["score"],
            }
            for idx, item in enumerate(ranking_info)
        ]
        return self.df, selected, eliminated, rankings

    def select_features_rfe(
        self,
        target_col: str,
        n_features_to_select: int = 5,
        task_type: str | None = None,
        feature_cols: list[str] | None = None,
        estimator_name: str = "Random Forest",
        step: float = 1.0,
    ) -> tuple[pd.DataFrame, list[str], list[str], list[dict[str, Any]]]:
        selected, eliminated, rankings, transformed_df = recursive_feature_elimination(
            self.df,
            target_col=target_col,
            feature_cols=feature_cols,
            task_type=task_type,
            n_features_to_select=n_features_to_select,
            step=step,
            estimator_name=estimator_name,
        )
        self.df = transformed_df
        return self.df, selected, eliminated, rankings

    def get_data(self) -> pd.DataFrame:
        return self.df


def compute_mutual_information(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Computes Mutual Information (MI) scores between feature columns and a target.
    Handles non-linear feature dependencies for classification and regression tasks.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    if not feature_cols:
        return []

    # Prepare clean data without nulls in target
    clean_df = df[[target_col] + feature_cols].dropna(subset=[target_col]).copy()
    if clean_df.empty:
        return [{"feature": f, "score": 0.0, "normalized_score": 0.0} for f in feature_cols]

    y_series = clean_df[target_col]
    if task_type is None:
        if not is_numeric_dtype(y_series) or y_series.nunique() <= 12:
            task_type = "classification"
        else:
            task_type = "regression"

    # Encode target if classification with string/categorical
    if task_type == "classification":
        if not is_numeric_dtype(y_series):
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_series.astype(str))
        else:
            y = y_series.values
    else:
        y = y_series.astype(float).values

    # Prepare features: encode categoricals and impute nulls
    X_mat = pd.DataFrame(index=clean_df.index)
    discrete_features_mask: list[bool] = []

    for col in feature_cols:
        series = clean_df[col]
        if not is_numeric_dtype(series) or series.dtype == "object":
            le = LabelEncoder()
            filled = series.fillna("missing").astype(str)
            X_mat[col] = le.fit_transform(filled)
            discrete_features_mask.append(True)
        else:
            median_val = series.median()
            X_mat[col] = series.fillna(median_val if not np.isnan(median_val) else 0)
            is_discrete = bool(series.nunique() < 10 and pd.api.types.is_integer_dtype(series))
            discrete_features_mask.append(is_discrete)

    if task_type == "classification":
        scores = mutual_info_classif(
            X_mat.values,
            y,
            discrete_features=discrete_features_mask,
            random_state=42,
        )
    else:
        scores = mutual_info_regression(
            X_mat.values,
            y,
            discrete_features=discrete_features_mask,
            random_state=42,
        )

    max_score = float(np.max(scores)) if len(scores) > 0 and np.max(scores) > 0 else 1.0
    results: list[dict[str, Any]] = []

    for col, score in zip(feature_cols, scores):
        val = float(score) if not np.isnan(score) else 0.0
        results.append({
            "feature": col,
            "score": round(val, 4),
            "normalized_score": round(min(1.0, val / max_score), 4) if max_score > 0 else 0.0,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def recursive_feature_elimination(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task_type: str | None = None,
    n_features_to_select: int = 5,
    step: float = 1.0,
    estimator_name: str = "Random Forest",
) -> tuple[list[str], list[str], list[dict[str, Any]], pd.DataFrame]:
    """
    Executes Recursive Feature Elimination (RFE) to prune redundant / collinear features.
    Returns: (selected_features, eliminated_features, rankings_list, transformed_dataframe)
    """
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    if not feature_cols:
        return [], [], [], df

    clean_df = df[[target_col] + feature_cols].dropna(subset=[target_col]).copy()
    if clean_df.empty:
        return feature_cols, [], [], df

    y_series = clean_df[target_col]
    if task_type is None:
        if not is_numeric_dtype(y_series) or y_series.nunique() <= 12:
            task_type = "classification"
        else:
            task_type = "regression"

    if task_type == "classification":
        if not is_numeric_dtype(y_series):
            le = LabelEncoder()
            y = le.fit_transform(y_series.astype(str))
        else:
            y = y_series.values
    else:
        y = y_series.astype(float).values

    X_mat = pd.DataFrame(index=clean_df.index)
    for col in feature_cols:
        series = clean_df[col]
        if not is_numeric_dtype(series) or series.dtype == "object":
            le = LabelEncoder()
            X_mat[col] = le.fit_transform(series.fillna("missing").astype(str))
        else:
            median_val = series.median()
            X_mat[col] = series.fillna(median_val if not np.isnan(median_val) else 0)

    # Determine estimator
    normalized_est = (estimator_name or "Random Forest").strip().lower()
    if task_type == "classification":
        if "gradient" in normalized_est:
            estimator = GradientBoostingClassifier(n_estimators=50, random_state=42)
        elif "logistic" in normalized_est or "linear" in normalized_est:
            estimator = LogisticRegression(max_iter=500, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        if "gradient" in normalized_est:
            estimator = GradientBoostingRegressor(n_estimators=50, random_state=42)
        elif "ridge" in normalized_est or "linear" in normalized_est:
            estimator = Ridge(random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)

    actual_n_select = max(1, min(n_features_to_select, len(feature_cols)))
    rfe_step = int(step) if step >= 1 else float(step)
    rfe = RFE(estimator=estimator, n_features_to_select=actual_n_select, step=rfe_step)
    rfe.fit(X_mat, y)

    selected_features = [col for col, sup in zip(feature_cols, rfe.support_) if sup]
    eliminated_features = [col for col, sup in zip(feature_cols, rfe.support_) if not sup]

    rankings: list[dict[str, Any]] = [
        {
            "feature": col,
            "ranking": int(rank),
            "selected": bool(sup),
            "score": None,
        }
        for col, rank, sup in zip(feature_cols, rfe.ranking_, rfe.support_)
    ]
    rankings.sort(key=lambda x: x["ranking"])

    # Create transformed dataframe keeping selected features and all other non-feature columns
    cols_to_keep = [c for c in df.columns if c not in eliminated_features]
    transformed_df = df[cols_to_keep].copy()

    return selected_features, eliminated_features, rankings, transformed_df
