from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.preprocess import (
    DropColumnsRequest,
    EncodeRequest,
    FeatureSelectionRequest,
    FeatureSelectionResponse,
    MissingValuesRequest,
    MutualInformationRequest,
    MutualInformationResponse,
    OutlierRequest,
    ScaleRequest,
)
from app.schemas.results import ProjectSnapshot
from app.services.dataset_service import dataset_store
from app.services.preprocessing_service import preprocessing_service


router = APIRouter(prefix="/preprocess", tags=["preprocess"])


def _get_session(project_id: str):
    try:
        return dataset_store.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/drop-columns", response_model=ProjectSnapshot)
def drop_columns(request: DropColumnsRequest) -> ProjectSnapshot:
    session = preprocessing_service.drop_columns(_get_session(request.project_id), request)
    return dataset_store.build_snapshot(session)


@router.post("/missing-values", response_model=ProjectSnapshot)
def missing_values(request: MissingValuesRequest) -> ProjectSnapshot:
    session = preprocessing_service.handle_missing_values(
        _get_session(request.project_id),
        request,
    )
    return dataset_store.build_snapshot(session)


@router.post("/encode", response_model=ProjectSnapshot)
def encode(request: EncodeRequest) -> ProjectSnapshot:
    session = preprocessing_service.encode(_get_session(request.project_id), request)
    return dataset_store.build_snapshot(session)


@router.post("/scale", response_model=ProjectSnapshot)
def scale(request: ScaleRequest) -> ProjectSnapshot:
    session = preprocessing_service.scale(_get_session(request.project_id), request)
    return dataset_store.build_snapshot(session)


@router.post("/outliers", response_model=ProjectSnapshot)
def outliers(request: OutlierRequest) -> ProjectSnapshot:
    session = preprocessing_service.handle_outliers(_get_session(request.project_id), request)
    return dataset_store.build_snapshot(session)


@router.post("/mutual-information", response_model=MutualInformationResponse)
def mutual_information(request: MutualInformationRequest) -> MutualInformationResponse:
    session = _get_session(request.project_id)
    return preprocessing_service.calculate_mutual_information(session, request)


@router.post("/feature-selection", response_model=FeatureSelectionResponse)
def feature_selection(request: FeatureSelectionRequest) -> FeatureSelectionResponse:
    session = _get_session(request.project_id)
    return preprocessing_service.apply_feature_selection(session, request)
