from pydantic import BaseModel


class PredictionRequest(BaseModel):

    Datetime: str

    Vehicle_Type: str

    Pickup_Location: str

    Drop_Location: str

    Booking_Value: float | None = None
    