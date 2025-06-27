import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for extracting time-based features from transactions
class TransactionFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df['TransactionWeekend'] = df['TransactionDayOfWeek'].isin([5, 6]).astype(int)
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        return df


# Aggregates customer-level statistics from transaction data
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date):
        self.reference_date = reference_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Group by CustomerId
        agg = df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (self.reference_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            AccountFrequency=('AccountId', pd.Series.nunique),
            Amount_sum=('Amount', 'sum'),
            Amount_mean=('Amount', 'mean'),
            Amount_std=('Amount', 'std'),
            Amount_min=('Amount', 'min'),
            Amount_max=('Amount', 'max'),
            Amount_count=('Amount', 'count'),
            Value_sum=('Value', 'sum'),
            Value_mean=('Value', 'mean'),
            AvgTransactionHour=('TransactionHour', 'mean'),
            MostFrequentDayOfWeek=('TransactionDayOfWeek', lambda x: x.mode().iloc[0] if not x.mode().empty else -1),
            CountryCode=('CountryCode', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
            CurrencyCode=('CurrencyCode', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
            ChannelId=('ChannelId', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        )

        agg.reset_index(inplace=True)
        return agg


# Builds transformation pipeline for categorical and numerical columns
def build_feature_pipeline(categorical_features, numerical_features):
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ('cat', cat_pipeline, categorical_features),
        ('num', num_pipeline, numerical_features)
    ])

    return full_pipeline


# Full preprocessing function
def preprocess_data(raw_df, reference_date):
    # Step 1: Feature extraction
    time_extractor = TransactionFeatureExtractor()
    time_features_df = time_extractor.fit_transform(raw_df)

    # Step 2: Aggregation per customer
    aggregator = CustomerAggregator(reference_date)
    customer_features_df = aggregator.fit_transform(time_features_df)

    # Step 3: Define feature columns
    categorical_features = ['CountryCode', 'CurrencyCode', 'ChannelId']
    numerical_features = [
         'Amount_sum', 'Amount_mean', 'Amount_std',
        'Amount_min', 'Amount_max', 'Amount_count', 'Value_sum', 'Value_mean']

    # Step 4: Apply pipeline
    pipeline = build_feature_pipeline(categorical_features, numerical_features)

    # Separate ID and features

    feature_data = customer_features_df[categorical_features + numerical_features]

    processed_array = pipeline.fit_transform(feature_data)
    feature_names = pipeline.get_feature_names_out()
    processed_df = pd.DataFrame(processed_array,columns=feature_names)

    # Reattach CustomerId
    extra_vars = customer_features_df[['Frequency','AvgTransactionHour', 'MostFrequentDayOfWeek','Recency','AccountFrequency']].reset_index(drop=True)
    customer_ids = customer_features_df[['CustomerId']].reset_index(drop=True)
    final_df = pd.concat([customer_ids, processed_df, extra_vars], axis=1)
    return final_df






















































