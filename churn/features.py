"""Feature pipeline shared by training and the API — mirrors the notebook's preprocessing."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BINARY = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'OnlineSecurity',
          'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
NOMINAL = ['MultipleLines', 'InternetService', 'PaymentMethod']
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']
CONTRACT = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
RAW_FIELDS = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
              'MonthlyCharges', 'TotalCharges']


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[RAW_FIELDS].copy()
    # 11 blank TotalCharges are tenure=0 customers, so 0 is the business-correct value
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0)
    # "No internet service" / "No phone service" count as No, as in the notebook
    df[BINARY] = (df[BINARY] == 'Yes').astype(int)
    df['Contract'] = df['Contract'].map(CONTRACT)
    # open-ended top bin so API inputs above the dataset max (8,685) still bin as High
    df['SpendingTier'] = pd.cut(df['TotalCharges'], [0, 1000, 4000, np.inf], labels=[0, 1, 2],
                                include_lowest=True).astype(int)
    df['TenureGroup'] = pd.cut(df['tenure'], [0, 12, 24, 48, 72], labels=[0, 1, 2, 3],
                               include_lowest=True).astype(int)
    return df


class Featurizer:
    """Fit on training rows only; the K-Means segment becomes a model feature."""

    def fit(self, raw: pd.DataFrame):
        df = clean(raw)
        self.ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore').fit(df[NOMINAL])
        base = self._encode(df)
        self.cluster_scaler = StandardScaler().fit(base)
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(self.cluster_scaler.transform(base))
        self.scaler = StandardScaler().fit(base[NUMERIC])
        return self

    def _encode(self, df):
        enc = pd.DataFrame(self.ohe.transform(df[NOMINAL]), columns=self.ohe.get_feature_names_out(NOMINAL),
                           index=df.index)
        return pd.concat([df.drop(columns=NOMINAL), enc], axis=1).astype(float)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        base = self._encode(clean(raw))
        out = base.copy()
        out['Cluster'] = self.kmeans.predict(self.cluster_scaler.transform(base))
        out[NUMERIC] = self.scaler.transform(base[NUMERIC])
        return out
