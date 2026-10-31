"""FastAPI churn-risk service (deployed on Azure App Service)."""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

BUNDLE = joblib.load(Path(__file__).parent / 'models' / 'churn_bundle.joblib')
YesNo = Literal['Yes', 'No']
AddOn = Literal['Yes', 'No', 'No internet service']
ADD_ONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

app = FastAPI(title='Telco Churn Risk API', version='1.0',
              description='XGBoost + SMOTE churn model on the IBM Telco dataset, recall-optimized.')


class Customer(BaseModel):
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: YesNo = 'No'
    Dependents: YesNo = 'No'
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: YesNo = 'Yes'
    MultipleLines: Literal['Yes', 'No', 'No phone service'] = 'No'
    InternetService: Literal['DSL', 'Fiber optic', 'No'] = 'Fiber optic'
    OnlineSecurity: AddOn = 'No'
    OnlineBackup: AddOn = 'No'
    DeviceProtection: AddOn = 'No'
    TechSupport: AddOn = 'No'
    StreamingTV: AddOn = 'No'
    StreamingMovies: AddOn = 'No'
    Contract: Literal['Month-to-month', 'One year', 'Two year'] = 'Month-to-month'
    PaperlessBilling: YesNo = 'Yes'
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                           'Credit card (automatic)'] = 'Electronic check'
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    @model_validator(mode='after')
    def consistent_services(self):
        no_net = self.InternetService == 'No'
        for name in ADD_ONS:
            if (getattr(self, name) == 'No internet service') != no_net:
                raise ValueError(f"{name} must be 'No internet service' exactly when InternetService is 'No'")
        if (self.MultipleLines == 'No phone service') != (self.PhoneService == 'No'):
            raise ValueError("MultipleLines must be 'No phone service' exactly when PhoneService is 'No'")
        return self


@app.get('/health')
def health():
    return {'status': 'ok', 'test_metrics': BUNDLE['metrics']}


@app.post('/predict')
def predict(c: Customer):
    X = BUNDLE['featurizer'].transform(pd.DataFrame([c.model_dump()]))[BUNDLE['columns']]
    # Recall-weighted score (SMOTE + scale_pos_weight), not a calibrated probability.
    # Test-set churn rate by band: high 53%, medium 19%, low 4% (base rate 26.5%).
    s = float(BUNDLE['model'].predict_proba(X)[0, 1])
    risk = 'high' if s >= 0.7 else 'medium' if s >= 0.4 else 'low'
    return {'churn_score': round(s, 3), 'risk': risk, 'kmeans_segment': int(X['Cluster'].iloc[0])}
