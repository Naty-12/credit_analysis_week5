import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from src.data_processing import build_feature_pipeline

def test_build_feature_pipeline():
    # Sample data
    df = pd.DataFrame({
        "CountryCode": ["ET", "US", np.nan],
        "CurrencyCode": ["ETB", "USD", "ETB"],
        "ChannelId": [1, 2, 3],
        "Amount_sum": [100, 200, 300],
        "Frequency": [1, 2, 3]
    })

    categorical_features = ["CountryCode", "CurrencyCode", "ChannelId"]
    numerical_features = ["Amount_sum", "Frequency"]

    pipeline = build_feature_pipeline(categorical_features, numerical_features)

    # Check pipeline type
    assert isinstance(pipeline, ColumnTransformer)

    # Check it transforms correctly
    transformed = pipeline.fit_transform(df)
    expected_num_cols = len(numerical_features)
    expected_cat_cols = 3 + 2 + 3  # ET/US/NaN for CountryCode, ETB/USD for CurrencyCode, 1/2/3 for ChannelId
    total_expected_cols = expected_num_cols + expected_cat_cols

    assert transformed.shape[1] == total_expected_cols