import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc


class DataVisualizer:
    """Collection of static Plotly-based visualization helpers."""

    @staticmethod
    def plot_missing_values(df: pd.DataFrame):
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            return None
        fig = px.bar(
            x=missing.index,
            y=missing.values,
            title="Missing Values per Column",
            labels={"x": "Column", "y": "Missing Count"},
            color=missing.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig

    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str):
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["Histogram + KDE", "Box Plot"]
        )
        fig.add_trace(
            go.Histogram(x=df[column], name="Distribution", nbinsx=30, opacity=0.75),
            row=1,
            col=1,
        )
        fig.add_trace(go.Box(y=df[column], name="Box Plot", boxmean=True), row=1, col=2)
        fig.update_layout(
            title_text=f"Distribution of <b>{column}</b>",
            showlegend=False,
            height=420,
        )
        return fig

    @staticmethod
    def plot_countplot(df: pd.DataFrame, column: str):
        counts = df[column].value_counts()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            title=f"Value Counts: <b>{column}</b>",
            labels={"x": column, "y": "Count"},
            color=counts.values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
        corr = numeric_df.corr().round(2)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.values,
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )
        fig.update_layout(title="Correlation Heatmap", height=600)
        return fig

    @staticmethod
    def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col=None):
        kwargs = dict(x=x_col, y=y_col, title=f"{x_col} vs {y_col}", opacity=0.7)
        if color_col:
            kwargs["color"] = color_col
        else:
            try:
                import statsmodels.api as sm  # noqa: F401
            except Exception:
                pass
            else:
                kwargs["trendline"] = "ols"
        fig = px.scatter(df, **kwargs)
        return fig

    @staticmethod
    def plot_scatter_matrix(df: pd.DataFrame, columns: list, color=None):
        dims = columns[: min(len(columns), 6)]
        fig = px.scatter_matrix(df, dimensions=dims, color=color, title="Scatter Matrix")
        fig.update_traces(diagonal_visible=False)
        return fig

    @staticmethod
    def plot_boxplot(df: pd.DataFrame, column: str, group_by=None):
        if group_by:
            fig = px.box(df, x=group_by, y=column, title=f"{column} by {group_by}")
        else:
            fig = px.box(df, y=column, title=f"Box Plot – {column}")
        return fig

    @staticmethod
    def plot_data_types(df: pd.DataFrame):
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(
            values=dtype_counts.values,
            names=[str(d) for d in dtype_counts.index],
            title="Data Types Distribution",
        )
        return fig

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, labels=None):
        if labels is None:
            labels = pd.Index(list(y_test) + list(y_pred)).drop_duplicates().tolist()
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        str_labels = [str(l) for l in labels]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=str_labels,
                y=str_labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14},
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=500,
        )
        return fig

    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba, classes):
        if len(classes) != 2:
            return None
        positive_label = classes[1]
        positive_index = list(classes).index(positive_label)
        fpr, tpr, _ = roc_curve(
            y_test,
            y_pred_proba[:, positive_index],
            pos_label=positive_label,
        )
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.3f})",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
        )
        return fig

    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred):
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred)
        min_v = min(y_test_arr.min(), y_pred_arr.min())
        max_v = max(y_test_arr.max(), y_pred_arr.max())
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_test_arr,
                y=y_pred_arr,
                mode="markers",
                name="Predictions",
                marker=dict(color="#1f77b4", opacity=0.6, size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="Actual vs Predicted",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            height=450,
        )
        return fig

    @staticmethod
    def plot_residuals(y_test, y_pred):
        residuals = np.array(y_test) - np.array(y_pred)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Residuals vs Predicted", "Residuals Distribution"],
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(y_pred),
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(color="#1f77b4", opacity=0.6, size=5),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(
            go.Histogram(x=residuals, name="Distribution", nbinsx=30, opacity=0.75),
            row=1,
            col=2,
        )
        fig.update_layout(title="Residual Analysis", showlegend=False, height=420)
        return fig

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
        top = importance_df.head(top_n)
        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {min(top_n, len(top))} Feature Importances",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(400, top_n * 22),
            coloraxis_showscale=False,
        )
        return fig

    @staticmethod
    def plot_cv_scores(cv_scores: np.ndarray, scoring_name: str = "Score"):
        folds = [f"Fold {i + 1}" for i in range(len(cv_scores))]
        mean_score = cv_scores.mean()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=folds,
                y=cv_scores,
                name=scoring_name,
                marker_color="#1f77b4",
                opacity=0.8,
            )
        )
        fig.add_hline(
            y=mean_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_score:.4f}",
        )
        fig.update_layout(
            title=f"Cross-Validation {scoring_name}s",
            xaxis_title="Fold",
            yaxis_title=scoring_name,
            height=380,
        )
        return fig

    @staticmethod
    def plot_mutual_information(scores_list: list[dict], target_col: str = "target"):
        if not scores_list:
            return None
        sorted_scores = sorted(scores_list, key=lambda x: x["score"], reverse=False)
        features = [item["feature"] for item in sorted_scores]
        scores = [item["score"] for item in sorted_scores]
        fig = go.Figure(
            data=go.Bar(
                x=scores,
                y=features,
                orientation="h",
                marker=dict(
                    color=scores,
                    colorscale="Teal",
                    showscale=True,
                    colorbar=dict(title="MI Score"),
                ),
                text=[f"{s:.4f}" for s in scores],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"Mutual Information Scores with Target <b>'{target_col}'</b>",
            xaxis_title="Mutual Information (Dependency Strength)",
            yaxis_title="Feature",
            height=max(360, len(features) * 26),
            margin=dict(l=140, r=40, t=50, b=50),
        )
        return fig

    @staticmethod
    def plot_optuna_history(trials_history: list[dict], metric_name: str = "Score"):
        if not trials_history:
            return None
        trial_nums = [t["trial_number"] for t in trials_history]
        trial_vals = [t.get("value") for t in trials_history]
        best_vals = [t.get("best_value") for t in trials_history]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trial_nums,
                y=trial_vals,
                mode="markers",
                name="Trial Score",
                marker=dict(size=9, color="#0ea5e9", opacity=0.75),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trial_nums,
                y=best_vals,
                mode="lines",
                name="Best Objective",
                line=dict(color="#10b981", width=3),
            )
        )
        fig.update_layout(
            title="Optuna Hyperparameter Optimization History",
            xaxis_title="Trial Number",
            yaxis_title=f"Objective ({metric_name})",
            height=400,
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def plot_optuna_param_importances(param_importances: dict[str, float]):
        if not param_importances:
            return None
        sorted_items = sorted(param_importances.items(), key=lambda x: x[1], reverse=False)
        params = [k for k, v in sorted_items]
        importances = [v for k, v in sorted_items]

        fig = go.Figure(
            data=go.Bar(
                x=importances,
                y=params,
                orientation="h",
                marker=dict(color="#6366f1"),
                text=[f"{v:.1%}" for v in importances],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Hyperparameter Importances",
            xaxis_title="Relative Importance",
            yaxis_title="Hyperparameter",
            height=max(320, len(params) * 32),
            margin=dict(l=140, r=40, t=50, b=50),
        )
        return fig

    @staticmethod
    def plot_shap_summary(shap_payload: dict):
        if not shap_payload or "beeswarm_points" not in shap_payload:
            return None

        points = shap_payload["beeswarm_points"]
        if not points:
            return None

        # Sort features by global importance
        feat_importance = {item["feature"]: item["importance"] for item in shap_payload.get("feature_importance", [])}
        sorted_features = sorted(feat_importance.keys(), key=lambda f: feat_importance[f], reverse=True)[:10]

        df_pts = pd.DataFrame(points)
        df_pts = df_pts[df_pts["feature"].isin(sorted_features)]

        fig = go.Figure()
        for f in reversed(sorted_features):
            sub = df_pts[df_pts["feature"] == f]
            if sub.empty:
                continue
            f_vals = sub["feature_value"].values
            min_v = f_vals.min() if len(f_vals) > 0 else 0
            max_v = f_vals.max() if len(f_vals) > 0 else 1
            norm_vals = (f_vals - min_v) / (max_v - min_v + 1e-8)

            # Add jitter on y
            jitter = np.random.normal(0, 0.08, size=len(sub))
            y_coords = [f] * len(sub)

            fig.add_trace(
                go.Scatter(
                    x=sub["shap_value"],
                    y=y_coords,
                    mode="markers",
                    name=f,
                    showlegend=False,
                    marker=dict(
                        size=7,
                        color=norm_vals,
                        colorscale="RdBu_r",
                        opacity=0.7,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=[f"Feature val: {v:.3f}<br>SHAP: {s:.3f}" for v, s in zip(f_vals, sub["shap_value"])],
                    hoverinfo="text",
                )
            )

        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig.update_layout(
            title="TreeSHAP Summary Distribution (Beeswarm)",
            xaxis_title="SHAP value (Impact on Model Output)",
            yaxis_title="Feature",
            height=max(420, len(sorted_features) * 38),
            margin=dict(l=140, r=40, t=50, b=50),
        )
        return fig

    @staticmethod
    def plot_shap_waterfall(local_explanation: dict, sample_index: int = 0):
        if not local_explanation:
            return None

        base_value = local_explanation.get("base_value", 0.0)
        output_value = local_explanation.get("output_value", 0.0)
        contributions = local_explanation.get("contributions", [])[:10]

        if not contributions:
            return None

        features = [c["feature"] for c in contributions]
        shap_vals = [c["shap_value"] for c in contributions]
        feature_vals = [c.get("value") for c in contributions]

        colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in shap_vals]

        fig = go.Figure(
            data=go.Bar(
                x=shap_vals,
                y=features,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:+.4f} (val: {fv})" for v, fv in zip(shap_vals, feature_vals)],
                textposition="outside",
            )
        )
        fig.add_vline(x=0, line_dash="solid", line_color="rgba(255,255,255,0.4)")
        fig.update_layout(
            title=f"SHAP Local Feature Contributions (Sample #{sample_index + 1}) · Base: {base_value:.3f} → Prediction: {output_value:.3f}",
            xaxis_title="SHAP Feature Contribution (+ pushes prediction up, - pushes down)",
            yaxis_title="Feature",
            height=max(360, len(features) * 32),
            margin=dict(l=140, r=60, t=60, b=50),
        )
        return fig

    @staticmethod
    def plot_shap_dependence(shap_payload: dict, feature_name: str):
        if not shap_payload or "beeswarm_points" not in shap_payload:
            return None

        points = [p for p in shap_payload["beeswarm_points"] if p["feature"] == feature_name]
        if not points:
            return None

        x_vals = [p["feature_value"] for p in points]
        y_vals = [p["shap_value"] for p in points]

        fig = go.Figure(
            data=go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=8, color="#06b6d4", opacity=0.75),
                text=[f"Value: {x:.3f}, SHAP: {y:.3f}" for x, y in zip(x_vals, y_vals)],
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig.update_layout(
            title=f"TreeSHAP Dependence: <b>{feature_name}</b>",
            xaxis_title=f"{feature_name} value",
            yaxis_title="SHAP value (Impact on Model Output)",
            height=380,
        )
        return fig
