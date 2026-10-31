"""Train XGBoost + SMOTE (recall-optimized grid search) and save the serving bundle."""
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from churn.features import Featurizer

ROOT = Path(__file__).resolve().parents[1]

raw = pd.read_csv(ROOT / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
y = (raw['Churn'] == 'Yes').astype(int)
raw_tr, raw_te, y_tr, y_te = train_test_split(raw, y, test_size=0.2, random_state=42, stratify=y)

feat = Featurizer().fit(raw_tr)
X_tr, X_te = feat.transform(raw_tr), feat.transform(raw_te)
X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=4),
    {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'scale_pos_weight': [1, 3]},
    cv=3, scoring='recall', n_jobs=2,
).fit(X_sm, y_sm)

model = grid.best_estimator_
proba = model.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    'recall': round(recall_score(y_te, pred), 4),
    'precision': round(precision_score(y_te, pred), 4),
    'roc_auc': round(roc_auc_score(y_te, proba), 4),
    'best_params': grid.best_params_,
}
print(json.dumps(metrics, indent=2))

(ROOT / 'models').mkdir(exist_ok=True)
joblib.dump({'featurizer': feat, 'model': model, 'columns': list(X_tr.columns), 'metrics': metrics},
            ROOT / 'models' / 'churn_bundle.joblib')
