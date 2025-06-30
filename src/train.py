import os  # For creating directories

import joblib  # To save/load preprocessors
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TransactionFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ensure 'TransactionStartTime' is datetime type for .dt accessor
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionDayOfWeek"] = df["TransactionStartTime"].dt.dayofweek
        df["TransactionWeekend"] = df["TransactionDayOfWeek"].isin([5, 6]).astype(int)
        df["TransactionYear"] = df["TransactionStartTime"].dt.year
        return df


# Aggregates customer-level statistics from transaction data
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date):
        self.reference_date = pd.Timestamp(reference_date)  # Ensure it's a Timestamp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Group by CustomerId
        agg = df.groupby("CustomerId").agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    (self.reference_date - x.max()).days if not x.empty else np.nan
                ),
            ),
            Frequency=("TransactionId", "count"),
            AccountFrequency=("AccountId", pd.Series.nunique),
            Amount_sum=("Amount", "sum"),
            Amount_mean=("Amount", "mean"),
            Amount_std=("Amount", "std"),
            Amount_min=("Amount", "min"),
            Amount_max=("Amount", "max"),
            Amount_count=("Amount", "count"),
            Value_sum=("Value", "sum"),
            Value_mean=("Value", "mean"),
            AvgTransactionHour=("TransactionHour", "mean"),
            MostFrequentDayOfWeek=(
                "TransactionDayOfWeek",
                lambda x: x.mode().iloc[0] if not x.mode().empty else -1,
            ),
            CountryCode=(
                "CountryCode",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
            CurrencyCode=(
                "CurrencyCode",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
            ChannelId=(
                "ChannelId",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
        )
        agg.reset_index(inplace=True)
        return agg


# Builds transformation pipeline for categorical and numerical columns
def build_feature_pipeline(categorical_features, numerical_features):
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    full_pipeline = ColumnTransformer(
        [
            ("cat", cat_pipeline, categorical_features),
            ("num", num_pipeline, numerical_features),
        ],
        remainder="passthrough",
    )  # Keep other columns if not transformed by these pipes

    return full_pipeline


# Full preprocessing function
def preprocess_data_and_save_pipeline(raw_df, reference_date):
    # Step 1: Feature extraction
    time_extractor = TransactionFeatureExtractor()
    time_features_df = time_extractor.fit_transform(raw_df)

    # Step 2: Aggregation per customer
    aggregator = CustomerAggregator(reference_date)
    customer_features_df = aggregator.fit_transform(time_features_df)

    # Step 3: Define feature columns
    categorical_features = ["CountryCode", "CurrencyCode", "ChannelId"]
    numerical_features = [
        "Amount_sum",
        "Amount_mean",
        "Amount_std",
        "Amount_min",
        "Amount_max",
        "Amount_count",
        "Value_sum",
        "Value_mean",
    ]

    # Step 4: Build and FIT the ColumnTransformer pipeline
    feature_transformer_pipeline = build_feature_pipeline(
        categorical_features, numerical_features
    )

    # Prepare data for the ColumnTransformer
    feature_data_for_transform = customer_features_df[
        categorical_features + numerical_features
    ]

    # Fit and transform the data using the ColumnTransformer
    processed_array = feature_transformer_pipeline.fit_transform(
        feature_data_for_transform
    )

    # Get feature names from the fitted ColumnTransformer
    # This will give names like 'cat__CountryCode_256', 'num__Amount_sum', etc.
    transformed_feature_names = feature_transformer_pipeline.get_feature_names_out()
    processed_df = pd.DataFrame(processed_array, columns=transformed_feature_names)

    # Reattach CustomerId and extra_vars
    extra_vars = customer_features_df[
        [
            "Frequency",
            "AvgTransactionHour",
            "MostFrequentDayOfWeek",
            "Recency",
            "AccountFrequency",
        ]
    ].reset_index(drop=True)
    customer_ids = customer_features_df[["CustomerId"]].reset_index(drop=True)

    # Final DataFrame for training
    final_df = pd.concat([customer_ids, processed_df, extra_vars], axis=1)

    # --- Save the fitted pipeline and final feature names ---
    models_dir = "models"
    os.makedirs(
        models_dir, exist_ok=True
    )  # Create 'models' directory if it doesn't exist

    joblib.dump(
        feature_transformer_pipeline,
        os.path.join(models_dir, "fitted_feature_pipeline.pkl"),
    )
    print(
        f"✅pipeline saved to {os.path.join(models_dir, 'fitted_feature_pipeline.pkl')}"
    )

    # Combine names from the transformed features and the extra_vars
    final_model_feature_names = (
        transformed_feature_names.tolist() + extra_vars.columns.tolist()
    )
    joblib.dump(
        final_model_feature_names,
        os.path.join(models_dir, "final_model_feature_names.pkl"),
    )
    print(
        f"✅names saved to {os.path.join(models_dir, 'final_model_feature_names.pkl')}"
    )
    # --------------------------------------------------------

    return (
        final_df,
        feature_transformer_pipeline,
    )  # Return the fitted pipeline along with the DataFrame


try:
    raw_transaction_data = pd.read_csv(
        "C:/Users/techin/credit_analysis_week5/data/raw/data.csv"
    )
    # Ensure TransactionStartTime is datetime for extraction
    raw_transaction_data["TransactionStartTime"] = pd.to_datetime(
        raw_transaction_data["TransactionStartTime"]
    )
    print("✅ Raw transaction data loaded.")
except FileNotFoundError:
    print(
        "Error: 'data/raw/raw_transactions.csv' not found. "
    )
    exit()  # Exit if raw data not found

# --- 3. Define your reference date for Recency ---

TRAINING_REFERENCE_DATE = raw_transaction_data["TransactionStartTime"].max()


# --- 4. Preprocess the data for training and save the fitted pipeline ---
print("🚀 Starting data preprocessing...")
processed_df_for_training, fitted_preprocessor_for_training = (
    preprocess_data_and_save_pipeline(raw_transaction_data, TRAINING_REFERENCE_DATE)
)
print("✅ Data preprocessing complete.")


try:
    # Load your labels (CustomerId and is_high_risk)
    labels_df = pd.read_csv(
        "C:/Users/techin/credit_analysis_week5/data/processed/cleaned_risk.csv"
    )  # Adjust path to your labels
    # Merge labels with the processed features
    merged_df = pd.merge(
        processed_df_for_training, labels_df, on="CustomerId", how="left"
    )
    # Handle potential NaNs if some customers are not labeled
    merged_df.dropna(subset=["is_high_risk"], inplace=True)
    merged_df["is_high_risk"] = merged_df["is_high_risk"].astype(
        int
    )  # Ensure target is int
    print("✅ Labels merged with processed data.")
except FileNotFoundError:
    print(
        "Error: 'data/processed/customer_labels.csv' not found."
    )
    # If `is_high_risk` is part of `raw_transaction_data` and needs to be aggregated:
    exit()

# Prepare features (X) and target (y) for model training
# Drop 'CustomerId' and 'is_high_risk' from features
X_train_final = merged_df.drop(columns=["CustomerId", "is_high_risk"])
y_train_final = merged_df["is_high_risk"]

# Verify that the column names in X_train_final match what your model expects
# This is why we saved `final_model_feature_names.pkl`
expected_feature_names = joblib.load("models/final_model_feature_names.pkl")
if not list(X_train_final.columns) == expected_feature_names:
    print("❗ Warning: Feature names mismatch before training!")
    print("Expected:", expected_feature_names)
    print("Actual:", list(X_train_final.columns))
    # You might need to reorder or align columns here if they differ.
    X_train_final = X_train_final[expected_feature_names]  # Force order/presence
else:
    print("✅ Feature names match expected order for training.")


# --- 6. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_train_final, y_train_final, stratify=y_train_final, test_size=0.2, random_state=42
)
print("✅ Data split into training and testing sets.")

# --- 7. Define and train your model ---
best_rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=2, random_state=42
)
print("🚀 Training RandomForestClassifier...")
best_rf.fit(X_train, y_train)
print("✅ RandomForestClassifier trained.")

# --- 8. Evaluate model ---
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
print("✅ Model evaluated.")
for metric_name, value in metrics.items():
    print(f"  {metric_name}: {value:.4f}")

# --- 9. Log to MLflow ---
print("🚀 Logging model to MLflow...")
with mlflow.start_run(run_name="RandomForest_FullPipeline_Model"):
    mlflow.log_params(best_rf.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="model",
        registered_model_name="credit_risk_model",
    )
print("✅ Model logged to MLflow successfully.")

# --- 10. Save model locally (optional) ---
joblib.dump(best_rf, "models/best_credit_risk_model.pkl")
print("✅ Best model saved locally to models/best_credit_risk_model.pkl")
