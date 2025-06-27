import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Load the labeled dataset with is_high_risk
df = pd.read_csv("data/processed/cleaned_data_with_risk.csv")

# Prepare features and target
X = df.drop(columns=["CustomerId", "is_high_risk"])
y = df["is_high_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define best model
best_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

# Train model
best_rf.fit(X_train, y_train)

# Evaluate model
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba)
}

# Log experiment in MLflow
with mlflow.start_run(run_name="RandomForest_BestModel"):
    mlflow.log_params(best_rf.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="model",
        registered_model_name="credit_risk_model"
    )

print("✅ Model trained, evaluated, and logged to MLflow successfully.")