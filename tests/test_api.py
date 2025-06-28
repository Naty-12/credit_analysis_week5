import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.api.main import app  # adjust import path if needed

@pytest.fixture(scope="module")
def client():
    # Use a context manager to ensure startup events run
    with TestClient(app) as c:
        yield c

def test_predict_single_transaction(client):
    # Test a valid prediction request returns 200 and expected keys
    response = client.post("/predict", json=[
        {
            "CustomerId": "C123",
            "TransactionId": "T1001",
            "TransactionStartTime": "2025-06-15T14:00:00",
            "CountryCode": "ET",
            "CurrencyCode": "ETB",
            "ChannelId": 1,
            "Amount": 200.5,
            "Value": 1000.0,
            "AccountId": "A55"
        }
    ])
    assert response.status_code == 200
    json_response = response.json()
    assert isinstance(json_response, list)
    assert "CustomerId" in json_response[0]
    assert "risk_probability" in json_response[0]

def test_predict_invalid_data(client):
    # Test missing required field 'Amount' results in 422 validation error
    response = client.post("/predict", json=[
        {
            "CustomerId": "C123",
            "TransactionId": "T1001",
            "TransactionStartTime": "2025-06-15T14:00:00",
            "CountryCode": "ET",
            "CurrencyCode": "ETB",
            "ChannelId": 1,
            "Value": 1000.0,
            "AccountId": "A55"
        }
    ])
    assert response.status_code == 422  # Unprocessable Entity (validation error)
