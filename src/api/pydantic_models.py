from datetime import datetime
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    CustomerId: str = Field(
        ..., example="C123", description="Unique identifier for the customer."
    )
    TransactionId: str = Field(
        ..., example="T1001", description="Unique identifier for the transaction."
    )
    # IMP: Use datetime.datetime for automatic ISO 8601 parsing and validation
    TransactionStartTime: datetime = Field(
        ...,
        example="2025-06-15T14:00:00",
        description="Start time of the transaction (ISO 8601 format).",
    )
    CountryCode: str = Field(
        ...,
        example="ET",
        description="ISO 3166-1 alpha-2 country code of the transaction.",
    )
    CurrencyCode: str = Field(
        ..., example="ETB", description="ISO 4217 currency code of the transaction."
    )
    ChannelId: int = Field(
        ...,
        example=1,
        description="Identifier for the transaction channel.",
    )
    Amount: float = Field(
        ..., example=200.5, gt=0, description="Transaction amount (must be positive)."
    )  # Added validation
    Value: float = Field(
        ..., example=1000.0, gt=0, description="Transaction value (must be positive)."
    )  # Added validation
    AccountId: str = Field(
        ...,
        example="A55",
        description="Account identifier involved in the transaction.",
    )

    # Optional: Add Pydantic config for extra strictness or aliases
    class Config:
        json_schema_extra = {
            "example": {
                "CustomerId": "C123",
                "TransactionId": "T1001",
                "TransactionStartTime": "2025-06-15T14:00:00",
                "CountryCode": "ET",
                "CurrencyCode": "ETB",
                "ChannelId": 1,
                "Amount": 200.5,
                "Value": 1000.0,
                "AccountId": "A55",
            }
        }
