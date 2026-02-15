from pydantic import BaseModel, Field
from typing import Literal, Annotated

class UserInput(BaseModel):

    # Numerical Features
    tenure: Annotated[int, Field(..., ge=0, description="Number of months the customer has stayed")]
    MonthlyCharges: Annotated[float, Field(..., ge=0, description="Monthly charges of the customer")]
    TotalCharges: Annotated[float, Field(..., ge=0, description="Total charges of the customer")]

    # Categorical Features
    gender: Annotated[
        Literal["Female", "Male"],
        Field(..., description="Customer gender")
    ]

    SeniorCitizen: Annotated[
        Literal[0, 1],
        Field(..., description="Whether the customer is a senior citizen (1) or not (0)")
    ]

    Partner: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Whether the customer has a partner")
    ]

    Dependents: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Whether the customer has dependents")
    ]

    PhoneService: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Whether the customer has phone service")
    ]

    MultipleLines: Annotated[
        Literal["No phone service", "No", "Yes"],
        Field(..., description="Multiple lines service status")
    ]

    InternetService: Annotated[
        Literal["DSL", "Fiber optic", "No"],
        Field(..., description="Type of internet service")
    ]

    OnlineSecurity: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(..., description="Online security service status")
    ]

    OnlineBackup: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(..., description="Online backup service status")
    ]

    DeviceProtection: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(..., description="Device protection service status")
    ]

    TechSupport: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(..., description="Tech support service status")
    ]

    StreamingTV: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(..., description="Streaming TV service status")
    ]

    StreamingMovies: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(..., description="Streaming movies service status")
    ]

    Contract: Annotated[
        Literal["Month-to-month", "One year", "Two year"],
        Field(..., description="Customer contract type")
    ]

    PaperlessBilling: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Whether customer uses paperless billing")
    ]

    PaymentMethod: Annotated[
        Literal[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        Field(..., description="Payment method of the customer")
    ]
