import traceback  # Import traceback for detailed error logging
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin

app = FastAPI()


# --- Custom Transformers from training ---
class TransactionFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract time-based features from 'TransactionStartTime'.
    Adds columns for hour, day, month, day of week, weekend indicator, and year.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ensure 'TransactionStartTime' is datetime type
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionDayOfWeek"] = df["TransactionStartTime"].dt.dayofweek
        # Convert boolean to integer (0 or 1) for 'TransactionWeekend'
        df["TransactionWeekend"] = df["TransactionDayOfWeek"].isin([5, 6]).astype(int)
        df["TransactionYear"] = df["TransactionStartTime"].dt.year
        return df


class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date):
        # Initialize with a reference date for recency calculation
        self.reference_date = pd.Timestamp(reference_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Group by CustomerId and apply various aggregations
        agg = df.groupby("CustomerId").agg(
            # Recency: days since last transaction
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    (self.reference_date - x.max()).days if not x.empty else np.nan
                ),
            ),
            # Frequency: count of transactions
            Frequency=("TransactionId", "count"),
            # AccountFrequency: number of unique accounts used
            AccountFrequency=("AccountId", pd.Series.nunique),
            # Aggregations for 'Amount'
            Amount_sum=("Amount", "sum"),
            Amount_mean=("Amount", "mean"),
            Amount_std=("Amount", "std"),
            Amount_min=("Amount", "min"),
            Amount_max=("Amount", "max"),
            Amount_count=("Amount", "count"),
            # Aggregations for 'Value'
            Value_sum=("Value", "sum"),
            Value_mean=("Value", "mean"),
            # Average transaction hour
            AvgTransactionHour=("TransactionHour", "mean"),
            # Most frequent day of week (handling empty mode result)
            MostFrequentDayOfWeek=(
                "TransactionDayOfWeek",
                lambda x: x.mode().iloc[0] if not x.mode().empty else -1,
            ),
            # Most frequent CountryCode (handling empty mode result)
            CountryCode=(
                "CountryCode",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
            # Most frequent CurrencyCode (handling empty mode result)
            CurrencyCode=(
                "CurrencyCode",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
            # Most frequent ChannelId (handling empty mode result)
            ChannelId=(
                "ChannelId",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
        )
        agg.reset_index(inplace=True)
        return agg
# --- Pydantic model for request validation and documentation ---


class Transaction(BaseModel):
    CustomerId: str = Field(
        ..., example="C123", description="Unique identifier for the customer."
    )
    TransactionId: str = Field(
        ..., example="T1001", description="Unique identifier for the transaction."
    )
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
    )
    Value: float = Field(
        ..., example=1000.0, gt=0, description="Transaction value (must be positive)."
    )
    AccountId: str = Field(
        ...,
        example="A55",
        description="Account identifier involved in the transaction.",
    )

    class Config:
        # Configuration for Pydantic model, including examples for API documentation
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


# --- Global variables to hold loaded models and feature names ---
# These are initialized to None and loaded during application startup.
loaded_model = None
feature_pipeline = None
expected_final_columns = None

# --- Feature lists used during model training and preprocessing ---
CATEGORICAL_FEATURES = ["CountryCode", "CurrencyCode", "ChannelId"]
NUMERICAL_FEATURES = [
    "Amount_sum",
    "Amount_mean",
    "Amount_std",
    "Amount_min",
    "Amount_max",
    "Amount_count",
    "Value_sum",
    "Value_mean",
]
EXTRA_VARS = [
    "Frequency",
    "AvgTransactionHour",
    "MostFrequentDayOfWeek",
    "Recency",
    "AccountFrequency",
]


# --- Application startup event handler ---
@app.on_event("startup")
async def load_models():
    """
    Loads the trained machine learning model, preprocessing pipeline,
    and expected final feature names into memory when the FastAPI application starts.
    This prevents reloading on every request and improves performance.
    """
    global loaded_model, feature_pipeline, expected_final_columns
    print("Attempting to load models...")
    try:
        loaded_model = joblib.load("models/best_credit_risk_model.pkl")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        loaded_model = None  # Ensure it's None if loading fails

    try:
        feature_pipeline = joblib.load("models/fitted_feature_pipeline.pkl")
        print("✅ Fitted preprocessing pipeline loaded.")
    except Exception as e:
        print(f"❌ Error loading feature pipeline: {e}")
        feature_pipeline = None  # Ensure it's None if loading fails

    try:
        expected_final_columns = joblib.load("models/final_model_feature_names.pkl")
        print("✅ Expected feature names loaded.")
    except Exception as e:
        print(f"❌ Error loading expected feature names: {e}")
        expected_final_columns = None  # Ensure it's None if loading fails
    print("Model loading attempt complete.")


# --- Inference endpoint ---
@app.post("/predict")
async def predict(transactions: List[Transaction]):
    """
    Receives a list of transaction data, preprocesses it,
    and returns predicted risk probabilities for each customer.
    """
    # Debugging: Check if all necessary components are loaded
    if not all([loaded_model, feature_pipeline, expected_final_columns]):
        print("DEBUG: Model, pipeline, or expected columns not loaded during request.")
        raise HTTPException(
            status_code=500,
            detail="Model or preprocessing pipeline not properly loaded.",
        )

    try:
        # Convert Pydantic models to a Pandas DataFrame
        # Use .model_dump() for Pydantic V2+, .dict() for V1
        raw_df = pd.DataFrame([t.model_dump() for t in transactions])
        raw_df["TransactionStartTime"] = pd.to_datetime(raw_df["TransactionStartTime"])

        print(f"DEBUG: Initial raw_df columns: {raw_df.columns.tolist()}")
        print(f"DEBUG: Initial raw_df dtypes:\n{raw_df.dtypes}")
        print(f"DEBUG: Initial raw_df head:\n{raw_df.head()}")

        # Use the current time as the reference date for recency calculation
        reference_date = pd.Timestamp.now()

        # Step 1: Extract time-based features using the custom transformer
        time_extractor = TransactionFeatureExtractor()
        time_features_df = time_extractor.transform(raw_df)
        print(f"DEBUG: time_features_df columns: {time_features_df.columns.tolist()}")
        print(f"DEBUG: time_features_df head:\n{time_features_df.head()}")

        # Step 2: Aggregate transaction data to customer level
        aggregator = CustomerAggregator(reference_date)
        customer_features_df = aggregator.transform(time_features_df)
        print(
            f"DEBUG: customer_features_df columns: "
            f"{customer_features_df.columns.tolist()}"
            )
        print(f"DEBUG: customer_features_df head:\n{customer_features_df.head()}")
        # Check for NaN in Amount_std after aggregation, especially for single rows
        if "Amount_std" in customer_features_df.columns:
            print(
                f"DEBUG: Amount_std after aggregation (first few values): "
                f"{customer_features_df['Amount_std'].head().tolist()}"
                )
            if customer_features_df["Amount_std"].isnull().any():
                print("DEBUG: WARNING! NaN found in 'Amount_std' after aggregation.")

        # Step 3: Prepare and apply the fitted feature pipeline
        # Select only the features that the ColumnTransformer expects
        cols_for_pipeline = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        # Ensure all columns expected by the pipeline exist, even if NaN
        transform_input = customer_features_df.reindex(
            columns=cols_for_pipeline, fill_value=np.nan
        )

        print(
            f"DEBUG: transform_input columns for pipeline:{transform_input.
                                                           columns.tolist()}"
        )
        print(f"DEBUG: transform_input dtypes:\n{transform_input.dtypes}")
        print(f"DEBUG: transform_input head:\n{transform_input.head()}")

        transformed_array = feature_pipeline.transform(transform_input)
        # Reconstruct DataFrame with correct column names from the pipeline
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=feature_pipeline.get_feature_names_out()
            )
        print(
            f"DEBUG: transformed_df columns after pipeline: {transformed_df.
                                                             columns.tolist()}"
        )
        print(f"DEBUG: transformed_df shape: {transformed_df.shape}")
        print(f"DEBUG: transformed_df head:\n{transformed_df.head()}")

        # Step 4: Add extra variables that are not part of the ColumnTransformer
        # Ensure indices align for concatenation
        extra_vars_df = (
            customer_features_df[EXTRA_VARS].reset_index(drop=True).fillna(0)
        )
        print(f"DEBUG: extra_vars_df columns: {extra_vars_df.columns.tolist()}")
        print(f"DEBUG: extra_vars_df head:\n{extra_vars_df.head()}")

        X_final = pd.concat([transformed_df, extra_vars_df], axis=1)
        print(f"DEBUG: X_final columns after concat: {X_final.columns.tolist()}")
        print(f"DEBUG: X_final shape after concat: {X_final.shape}")

        # Step 5: Ensure final column alignment with the m
        # This is critical for model prediction stability
        missing_cols = set(expected_final_columns) - set(X_final.columns)
        for col in missing_cols:
            X_final[col] = 0.0
        # Ensure columns are in the exact order expected by the model
        X_final = X_final[expected_final_columns]
        print(
            f"DEBUG: X_final columns before pred:{X_final.columns.tolist()}"
        )
        print(f"DEBUG: X_final shape before prediction: {X_final.shape}")
        print(f"DEBUG: X_final head before prediction:\n{X_final.head()}")

        # Step 6: Predict risk probabilities using the loaded model
        # For classification, predict_proba returns probabilities for each class
        # We usually take the probability of the positive class (index 1)
        risk_probs = loaded_model.predict_proba(X_final)[:, 1]
        print(f"DEBUG: Predictions calculated: {risk_probs.tolist()}")

        # Step 7: Format the output as a list of dictionaries, one per customer
        results = []
        for i, cid in enumerate(customer_features_df["CustomerId"]):
            results.append(
                {
                    "CustomerId": cid,
                    "risk_probability": float(
                        risk_probs[i]
                    ),  # Convert numpy float to Python float
                }
            )

        print("DEBUG: Prediction process complete. Returning results.")
        return results

    except Exception as e:
        # Catch any exceptions during the prediction process and return a 500 error
        print(f"ERROR: An exception occurred during prediction: {e}")
        traceback.print_exc()  # Print the full traceback to the console for debugging
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error during prediction: {e}"
        )
