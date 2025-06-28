## Project Overview

This project aims to develop a robust credit scoring system for Bati Bank’s new buy-now-pay-later service in partnership with an eCommerce platform. By leveraging customer behavioral data, we will predict credit risk, assign credit scores, and recommend optimal loan terms, all while ensuring compliance with Basel II regulatory standards.

## Objectives

- Define a proxy credit risk variable to classify customers as high risk or low risk using behavioral RFM data.
- Select and validate observable features that strongly predict the proxy credit risk variable.
- Develop a model to estimate the probability of default (risk probability) for new customers.
- Transform the risk probability into a standardized credit score to assess creditworthiness.

## Project Structure
``` 
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
