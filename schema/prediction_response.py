from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    predicted_label: str = Field(
        ...,
        description="Final churn prediction result",
        example="Churn"
    )

    churn_probability: float = Field(
        ...,
        description="Probability that the customer will churn (class 1)",
        example=0.82
    )

    no_churn_probability: float = Field(
        ...,
        description="Probability that the customer will not churn (class 0)",
        example=0.18
    )

    confidence: float = Field(
        ...,
        description="Confidence score of the predicted class",
        example=0.82
    )
