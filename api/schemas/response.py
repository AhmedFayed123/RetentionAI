"""Response schemas for the production API layer."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response payload returned after a churn prediction request."""

    prediction: int = Field(
        ...,
        description="Predicted churn class as an integer label.",
        ge=0,
        json_schema_extra={"example": 1},
    )
    label: str = Field(
        ...,
        description="Human-readable label for the predicted class.",
        json_schema_extra={"example": "Churn"},
    )
    probability: float = Field(
        ...,
        description="Probability of the predicted class.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.87},
    )
    model_name: str = Field(
        ...,
        description="Name of the model used to generate the prediction.",
        json_schema_extra={"example": "Logistic Regression"},
    )
