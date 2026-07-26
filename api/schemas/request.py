"""Request schemas for the production API layer."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Incoming customer payload for churn prediction requests."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    customerID: Optional[str] = Field(
        default=None,
        description="Unique customer identifier from the Telco churn dataset. "
        "Optional — used only for tracing/logging, not for prediction.",
        min_length=1,
        json_schema_extra={"example": "7590-VHVEG"},
    )
    gender: str = Field(
        ...,
        description="Customer gender.",
        min_length=1,
        json_schema_extra={"example": "Female"},
    )
    SeniorCitizen: int = Field(
        ...,
        description="Indicates whether the customer is a senior citizen (0 = No, 1 = Yes).",
        ge=0,
        le=1,
        json_schema_extra={"example": 0},
    )
    Partner: str = Field(
        ...,
        description="Whether the customer has a partner.",
        min_length=1,
        json_schema_extra={"example": "Yes"},
    )
    Dependents: str = Field(
        ...,
        description="Whether the customer has dependents.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
    tenure: int = Field(
        ...,
        description="Number of months the customer has stayed with the service provider.",
        ge=0,
        json_schema_extra={"example": 1},
    )
    PhoneService: str = Field(
        ...,
        description="Whether the customer has phone service.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
    MultipleLines: str = Field(
        ...,
        description="Whether the customer has multiple lines on the account.",
        min_length=1,
        json_schema_extra={"example": "No phone service"},
    )
    InternetService: str = Field(
        ...,
        description="The customer's internet service type.",
        min_length=1,
        json_schema_extra={"example": "DSL"},
    )
    OnlineSecurity: str = Field(
        ...,
        description="Whether the customer has online security enabled.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
    OnlineBackup: str = Field(
        ...,
        description="Whether the customer has online backup enabled.",
        min_length=1,
        json_schema_extra={"example": "Yes"},
    )
    DeviceProtection: str = Field(
        ...,
        description="Whether the customer has device protection enabled.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
    TechSupport: str = Field(
        ...,
        description="Whether the customer has technical support enabled.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
    StreamingTV: str = Field(
        ...,
        description="Whether the customer uses streaming TV service.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
    StreamingMovies: str = Field(
        ...,
        description="Whether the customer uses streaming movies service.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
    Contract: str = Field(
        ...,
        description="Type of customer contract.",
        min_length=1,
        json_schema_extra={"example": "Month-to-month"},
    )
    PaperlessBilling: str = Field(
        ...,
        description="Whether the customer uses paperless billing.",
        min_length=1,
        json_schema_extra={"example": "Yes"},
    )
    PaymentMethod: str = Field(
        ...,
        description="Customer payment method.",
        min_length=1,
        json_schema_extra={"example": "Electronic check"},
    )
    MonthlyCharges: float = Field(
        ...,
        description="Monthly charges billed to the customer.",
        ge=0,
        json_schema_extra={"example": 29.85},
    )
    TotalCharges: float = Field(
        ...,
        description="Total charges accumulated for the customer.",
        ge=0,
        json_schema_extra={"example": 29.85},
    )
    Churn: Optional[str] = Field(
        default=None,
        description="Historical churn label for the customer record. "
        "Optional — not required for prediction; provided only for evaluation.",
        min_length=1,
        json_schema_extra={"example": "No"},
    )
