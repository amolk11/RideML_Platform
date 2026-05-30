from fastapi import APIRouter

from app.schemas.request import (
    PredictionRequest
)

from app.schemas.response import (
    PredictionResponse
)

from app.services.prediction_service import (
    predict
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_endpoint(
    request: PredictionRequest
):

    return predict(
        request.model_dump()
    )