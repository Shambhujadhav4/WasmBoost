from __future__ import annotations

import json
from typing import Any

import plotly.io as pio
from modules.visualizations import DataVisualizer

from app.services.dataset_service import ProjectSession


class VisualizationService:
    def feature_importance(self, session: ProjectSession) -> list[dict[str, Any]] | None:
        if session.model_trainer is None:
            return None

        importance = session.model_trainer.get_feature_importance()
        if importance is None:
            return None
        return importance.to_dict(orient="records")

    def available_numeric_columns(self, session: ProjectSession) -> list[str]:
        return session.processed_data.select_dtypes(include=["number"]).columns.tolist()

    def available_categorical_columns(self, session: ProjectSession) -> list[str]:
        return session.processed_data.select_dtypes(include=["object", "category"]).columns.tolist()

    def _get_plot_data(self, session: ProjectSession):
        df = session.processed_data
        if len(df) > 2000:
            return df.sample(n=2000, random_state=42)
        return df

    def build_missing_values_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        figure = DataVisualizer.plot_missing_values(self._get_plot_data(session))
        return self._serialize_figure(figure)

    def build_correlation_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        figure = DataVisualizer.plot_correlation_heatmap(self._get_plot_data(session))
        return self._serialize_figure(figure)

    def build_distribution_figure(
        self,
        session: ProjectSession,
        column: str,
    ) -> dict[str, Any] | None:
        if column not in session.processed_data.columns:
            return None
        figure = DataVisualizer.plot_distribution(self._get_plot_data(session), column)
        return self._serialize_figure(figure)

    def build_boxplot_figure(
        self,
        session: ProjectSession,
        column: str,
    ) -> dict[str, Any] | None:
        if column not in session.processed_data.columns:
            return None
        figure = DataVisualizer.plot_boxplot(self._get_plot_data(session), column)
        return self._serialize_figure(figure)

    def build_countplot_figure(
        self,
        session: ProjectSession,
        column: str,
    ) -> dict[str, Any] | None:
        if column not in session.processed_data.columns:
            return None
        figure = DataVisualizer.plot_countplot(self._get_plot_data(session), column)
        return self._serialize_figure(figure)

    def build_scatter_figure(
        self,
        session: ProjectSession,
        x_col: str,
        y_col: str,
        color_col: str | None = None,
    ) -> dict[str, Any] | None:
        figure = DataVisualizer.plot_scatter(
            self._get_plot_data(session),
            x_col,
            y_col,
            color_col=color_col,
        )
        return self._serialize_figure(figure)

    def build_confusion_matrix_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        trainer = session.model_trainer
        if trainer is None or trainer.task_type != "classification":
            return None
        figure = DataVisualizer.plot_confusion_matrix(
            trainer.y_test,
            trainer.y_pred,
            labels=getattr(trainer.model, "classes_", None),
        )
        return self._serialize_figure(figure)

    def build_roc_curve_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        trainer = session.model_trainer
        if (
            trainer is None
            or trainer.task_type != "classification"
            or trainer.y_pred_proba is None
        ):
            return None
        classes = list(getattr(trainer.model, "classes_", trainer.y_test.unique().tolist()))
        figure = DataVisualizer.plot_roc_curve(
            trainer.y_test,
            trainer.y_pred_proba,
            classes,
        )
        return self._serialize_figure(figure)

    def build_actual_vs_predicted_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        trainer = session.model_trainer
        if trainer is None or trainer.task_type != "regression":
            return None
        figure = DataVisualizer.plot_actual_vs_predicted(
            trainer.y_test,
            trainer.y_pred,
        )
        return self._serialize_figure(figure)

    def build_residuals_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        trainer = session.model_trainer
        if trainer is None or trainer.task_type != "regression":
            return None
        figure = DataVisualizer.plot_residuals(trainer.y_test, trainer.y_pred)
        return self._serialize_figure(figure)

    def build_optuna_history_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        if not session.model_results or "optuna_optimization" not in session.model_results:
            return None
        opt_data = session.model_results["optuna_optimization"]
        figure = DataVisualizer.plot_optuna_history(
            opt_data.get("trials_history", []),
            metric_name=opt_data.get("metric_name", "Score"),
        )
        return self._serialize_figure(figure)

    def build_optuna_param_importances_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        if not session.model_results or "optuna_optimization" not in session.model_results:
            return None
        opt_data = session.model_results["optuna_optimization"]
        figure = DataVisualizer.plot_optuna_param_importances(
            opt_data.get("param_importances", {}),
        )
        return self._serialize_figure(figure)

    def build_shap_summary_figure(
        self,
        session: ProjectSession,
    ) -> dict[str, Any] | None:
        if not session.model_results or "shap_explanations" not in session.model_results:
            return None
        shap_payload = session.model_results["shap_explanations"]
        figure = DataVisualizer.plot_shap_summary(shap_payload)
        return self._serialize_figure(figure)

    def build_shap_waterfall_figure(
        self,
        session: ProjectSession,
        sample_index: int = 0,
    ) -> dict[str, Any] | None:
        if not session.model_results or "shap_explanations" not in session.model_results:
            return None
        shap_payload = session.model_results["shap_explanations"]
        explanations = shap_payload.get("sample_explanations", [])
        if not explanations:
            return None
        idx = max(0, min(sample_index, len(explanations) - 1))
        figure = DataVisualizer.plot_shap_waterfall(explanations[idx], sample_index=idx)
        return self._serialize_figure(figure)

    def build_shap_dependence_figure(
        self,
        session: ProjectSession,
        feature_name: str,
    ) -> dict[str, Any] | None:
        if not session.model_results or "shap_explanations" not in session.model_results:
            return None
        shap_payload = session.model_results["shap_explanations"]
        figure = DataVisualizer.plot_shap_dependence(shap_payload, feature_name)
        return self._serialize_figure(figure)

    def _serialize_figure(self, figure: Any) -> dict[str, Any] | None:
        if figure is None:
            return None
        return json.loads(pio.to_json(figure))


visualization_service = VisualizationService()
