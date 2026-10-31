# Customer Churn Prediction — IBM Telco

End-to-end ML pipeline that predicts telecom customer churn 30 days in advance, built from raw
data through EDA, feature engineering, unsupervised segmentation, and four tuned classifiers.

**Headline result:** XGBoost + SMOTE reaches **0.91 recall** (ROC AUC 0.74) — catching 9 of every
10 customers who would have churned, up from 0.56 recall before class-imbalance handling.

---

## Why recall, not accuracy

The business loses ~30% of customers annually, and acquiring a replacement costs 5–25x more than
retaining an existing customer. A wasted retention offer costs little; a missed churner costs their
entire future revenue. Every modeling decision — the scoring metric in Grid Search, SMOTE, the final
model recommendation — optimizes for **recall** on the churn class.

---

## Dataset

| Property | Value |
|---|---|
| Source | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) |
| Rows | 7,043 customers |
| Features | 21 |
| Churn rate | 26.54% (imbalanced) |

The `Churn` column is separated and held out as ground truth before any processing, simulating a
real-world **unlabeled** dataset. The pipeline reconstructs churn risk from scratch — clustering
runs without ever seeing the labels.

A copy of the dataset ships in [`data/`](data/) so the notebook is reproducible.

---

## Pipeline

```
Raw data → EDA → Preprocessing → Feature Engineering → K-Means Segmentation
        → Model Training → Grid Search (CV) → Evaluation → Business Report
```

### Preprocessing decisions

| Issue | Column | Fix |
|---|---|---|
| Stored as object | `TotalCharges` | Cast to float |
| 11 hidden missing values (`" "`) | `TotalCharges` | Filled with 0 — these are `tenure=0` new customers, so business logic set the value, not statistical imputation |
| Identifier | `customerID` | Dropped |
| Near-zero correlation | `gender` | Dropped |

Encoding is split by variable type — label encoding for binary flags, ordinal for `Contract`
(month-to-month < one year < two year), one-hot (drop-first) for true nominals.

**Leakage prevention:** `StandardScaler` is fitted on the training split only, then applied to test.
The 80/20 split is stratified to preserve the 26.54% churn rate in both halves.

### Feature engineering

`TotalCharges` correlates 0.83 with `tenure` and 0.65 with `MonthlyCharges` — a multicollinearity
problem. Two binned features replace that redundancy with interpretable business segments:

- **SpendingTier** — Low (`$0–1k`) / Mid (`$1k–4k`) / High (`$4k+`)
- **TenureGroup** — New (0–12mo) / Developing (12–24) / Established (24–48) / Loyal (48–72)

### Segmentation (K-Means, k=3 via elbow method)

| Cluster | Segment | Size | Avg tenure | Avg monthly | Churn rate |
|---|---|---|---|---|---|
| 0 | Loyal High Value | 1,648 | 62 mo | $94.50 | 15% |
| 1 | Mid Value Developing | 2,780 | 39 mo | $58.79 | 19% |
| 2 | New Low Value | 2,615 | 7 mo | $52.36 | **42%** |

Cluster membership is then fed back in as a feature for the supervised models.

---

## Results

Grid Search with 3-fold CV, scored on recall:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| SVM (tuned) | 0.781 | 0.593 | 0.556 | 0.574 | 0.709 |
| Logistic Regression (tuned) | 0.803 | 0.657 | 0.543 | 0.594 | 0.720 |
| K-Means Classifier | 0.729 | 0.492 | 0.671 | 0.568 | 0.710 |
| **XGBoost + SMOTE** | 0.662 | 0.434 | **0.906** | 0.587 | **0.740** |

SMOTE was applied to the **training set only**, balancing 1,495 churners against 4,139 non-churners.
That single change moved recall from 0.56 → 0.91.

**Which model to ship depends on intervention cost:**
- *Broad retention campaigns* → **XGBoost + SMOTE**. Catches 90.6% of churners; false alarms are cheap.
- *High-touch interventions* (account manager assignment) → **Logistic Regression**. Best precision (0.66) and F1.

---

## Business insights

1. **The first year is the danger zone.** Customers under 12 months tenure churn at 42% — ~3x the
   loyal segment. Found through unsupervised clustering, without labels.
2. **Contract type is the strongest signal.** Month-to-month customers leave far more often; no
   financial penalty means no friction to leave.
3. **Higher charges correlate with higher churn.** Price dissatisfaction is measurable.
4. **Fiber optic churns more than DSL** despite being the premium tier — suggests unmet expectations
   or competitor pricing pressure.

---

## Known limitations

| Limitation | Impact | Planned fix |
|---|---|---|
| XGBoost precision is 0.43 | 442 false positives | Tune the decision threshold from 0.5 → 0.6 |
| No complaint / service-call data | Missing the strongest behavioral signal | Integrate CRM complaint history |
| Static snapshot, no time series | No usage-trend features | Month-over-month usage deltas |
| No competitor data | External churn drivers unmodeled | Enrich with market pricing |

---

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/customer_churn_prediction.ipynb
```

The notebook was authored in Google Colab and reads the dataset from `/content/`. Running locally,
point that `read_csv` call at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`; in Colab, upload the file
from `data/` first.

---

## Repo layout

```
notebooks/customer_churn_prediction.ipynb   Full pipeline + embedded outputs and the analytics report
data/WA_Fn-UseC_-Telco-Customer-Churn.csv   IBM Telco dataset (7,043 rows)
requirements.txt
```

---

Built by **Omar Diaa** as part of the Digital Egypt Pioneers Initiative (DEPI) — AI & Data Science,
Microsoft Machine Learning Engineer track.

---

## Serving API

The model is retrained from a script (`python -m churn.train`) and served by a FastAPI app,
containerized with Docker (`Dockerfile`). A Streamlit app (`streamlit_app.py`) wraps the model and the
care agent for the live demo on Streamlit Community Cloud.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status plus held-out test metrics |
| `POST /predict` | Recall-weighted `churn_score`, `risk` band (low / medium / high), K-Means segment |
| `GET /docs` | Interactive API docs |

The training script fits every encoder, scaler and the K-Means segmenter on the training split only.
Held-out results: **recall 0.904, ROC AUC 0.842**. The score is recall-weighted (SMOTE +
`scale_pos_weight`), not a calibrated probability: test churn rate is 53% in the high band,
19% in medium and 4% in low (base rate 26.5%).

```bash
pip install -r requirements.txt
uvicorn app:app --reload            # API
streamlit run streamlit_app.py      # demo UI (scoring + care agent)
```

## Customer-Care Agent

`agent/care_agent.py` is a bilingual (Egyptian Arabic / English) LangGraph agent on Groq Llama 3.3
that answers customer questions with three tools:

- **run_sql** — read-only SQL over the customer table (enforced by SQLite `query_only`)
- **churn_risk** — calls the deployed churn API for the customer's risk band
- **search_knowledge** — retrieval over a sample plans / retention-offer / billing knowledge base

It checks churn risk before proposing a retention offer and prints the tool trace behind each answer.
The knowledge base is sample content written for this demo, not a real operator's plans or prices.

```bash
export GROQ_API_KEY=...            # required
export CHURN_API_URL=https://...   # optional; falls back to the local model
python -m agent.care_agent "Customer 7590-VHVEG wants to cancel. What can we offer?"
```
