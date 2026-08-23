from app.schemas.preprocess import (
    DropColumnsRequest,
    EncodeRequest,
    FeatureRankingItem,
    FeatureSelectionRequest,
    FeatureSelectionResponse,
    MissingValuesRequest,
    MutualInformationItem,
    MutualInformationRequest,
    MutualInformationResponse,
    OutlierRequest,
    ScaleRequest,
)
from app.services.dataset_service import ProjectSession, dataset_store
from modules.preprocessing import DataPreprocessor, compute_mutual_information
from modules.visualizations import DataVisualizer


class PreprocessingService:
    def drop_columns(self, session: ProjectSession, request: DropColumnsRequest) -> ProjectSession:
        processor = DataPreprocessor(session.processed_data)
        session.processed_data = processor.drop_columns(request.columns)
        session.preprocessing_log.append(f"Dropped columns: {request.columns}")
        dataset_store.clear_model_state(session)
        dataset_store.save_project(session)
        return session

    def handle_missing_values(
        self,
        session: ProjectSession,
        request: MissingValuesRequest,
    ) -> ProjectSession:
        processor = DataPreprocessor(session.processed_data)
        columns = request.columns or None
        session.processed_data = processor.handle_missing_values(
            request.strategy,
            columns,
            request.fill_value,
        )
        session.preprocessing_log.append(
            f"Missing values -> {request.strategy} on {columns or 'all columns'}"
        )
        dataset_store.clear_model_state(session)
        dataset_store.save_project(session)
        return session

    def encode(self, session: ProjectSession, request: EncodeRequest) -> ProjectSession:
        processor = DataPreprocessor(session.processed_data)
        columns = request.columns or None
        session.processed_data = processor.encode_categorical(request.method, columns)
        session.preprocessing_log.append(
            f"Encoding -> {request.method} on {columns or 'all categorical'}"
        )
        dataset_store.clear_model_state(session)
        dataset_store.save_project(session)
        return session

    def scale(self, session: ProjectSession, request: ScaleRequest) -> ProjectSession:
        processor = DataPreprocessor(session.processed_data)
        columns = request.columns or None
        session.processed_data = processor.scale_features(request.method, columns)
        session.preprocessing_log.append(
            f"Scaling -> {request.method} on {columns or 'all numeric'}"
        )
        dataset_store.clear_model_state(session)
        dataset_store.save_project(session)
        return session

    def handle_outliers(self, session: ProjectSession, request: OutlierRequest) -> ProjectSession:
        processor = DataPreprocessor(session.processed_data)
        columns = request.columns or None
        session.processed_data = processor.handle_outliers(
            request.method,
            columns,
            request.threshold,
        )
        session.preprocessing_log.append(
            f"Outliers -> {request.method} on {columns or 'all numeric'}"
        )
        dataset_store.clear_model_state(session)
        dataset_store.save_project(session)
        return session

    def calculate_mutual_information(
        self,
        session: ProjectSession,
        request: MutualInformationRequest,
    ) -> MutualInformationResponse:
        target = request.target_column
        feature_cols = (
            request.feature_columns
            if request.feature_columns
            else [c for c in session.processed_data.columns if c != target]
        )
        scores_list = compute_mutual_information(
            session.processed_data,
            target_col=target,
            feature_cols=feature_cols,
            task_type=request.task_type,
        )

        inferred_task = request.task_type or (
            "classification"
            if not pd.api.types.is_numeric_dtype(session.processed_data[target])
            or session.processed_data[target].nunique() <= 12
            else "regression"
        )

        import plotly.io as pio
        fig = DataVisualizer.plot_mutual_information(scores_list, target_col=target)
        fig_json = json.loads(pio.to_json(fig)) if fig is not None else None

        return MutualInformationResponse(
            project_id=session.project_id,
            target_column=target,
            task_type=inferred_task,
            scores=[MutualInformationItem(**item) for item in scores_list],
            figure=fig_json,
        )

    def apply_feature_selection(
        self,
        session: ProjectSession,
        request: FeatureSelectionRequest,
    ) -> FeatureSelectionResponse:
        processor = DataPreprocessor(session.processed_data)
        target = request.target_column
        features = request.feature_columns or [c for c in session.processed_data.columns if c != target]

        if request.method == "mi":
            updated_df, selected, eliminated, rankings = processor.select_features_mi(
                target_col=target,
                n_features_to_select=request.n_features_to_select,
                task_type=request.task_type,
                feature_cols=features,
            )
            session.preprocessing_log.append(
                f"Feature Selection (Mutual Information) -> Kept {len(selected)} top features, pruned {len(eliminated)}: {eliminated}"
            )
        else:
            updated_df, selected, eliminated, rankings = processor.select_features_rfe(
                target_col=target,
                n_features_to_select=request.n_features_to_select,
                task_type=request.task_type,
                feature_cols=features,
                estimator_name=request.rfe_estimator or "Random Forest",
                step=request.step,
            )
            session.preprocessing_log.append(
                f"Feature Selection (RFE with {request.rfe_estimator}) -> Kept {len(selected)} features, pruned {len(eliminated)}: {eliminated}"
            )

        session.processed_data = updated_df
        dataset_store.clear_model_state(session)
        dataset_store.save_project(session)

        snapshot_dict = dataset_store.build_snapshot(session).model_dump()

        return FeatureSelectionResponse(
            project_id=session.project_id,
            method=request.method,
            target_column=target,
            selected_features=selected,
            eliminated_features=eliminated,
            rankings=[FeatureRankingItem(**r) for r in rankings],
            snapshot=snapshot_dict,
        )


import json
import pandas as pd

preprocessing_service = PreprocessingService()
