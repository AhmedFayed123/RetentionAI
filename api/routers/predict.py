"""Prediction router for the RetentionAI API."""

from fastapi import APIRouter

from api.schemas.request import PredictionRequest
from api.schemas.response import PredictionResponse
from api.services.prediction_service import PredictionService

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Orchestrate prediction requests through the service layer."""
    service = PredictionService()
    return service.predict(request)


@router.get("/predict/test")
def predict_test() -> dict[str, str]:
    """Temporary debug endpoint to verify the prediction router is registered."""
    return {"status": "ok"}
