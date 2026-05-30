from pydantic import BaseModel


class PredictionResponse(BaseModel):

    prediction: int

    probability: float
    