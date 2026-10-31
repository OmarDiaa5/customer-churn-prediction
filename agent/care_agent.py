"""Bilingual (Egyptian Arabic / English) telco customer-care agent.

LangGraph ReAct agent on Groq Llama 3.3 with three tools:
  - run_sql:          read-only SQL over the IBM Telco customer table (SQLite)
  - churn_risk:       calls the deployed churn API (falls back to the local model bundle)
  - search_knowledge: retrieval over the care knowledge base (char n-gram TF-IDF, works for Arabic)
Every answer prints the tool trace that produced it.

Usage:  python -m agent.care_agent "Customer 7590-VHVEG wants to cancel, what do I offer?"
Env:    GROQ_API_KEY (required), CHURN_API_URL (optional, e.g. https://<app>.azurewebsites.net)
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import httpx
import pandas as pd
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# --- data: the Telco table in an in-memory SQLite DB (Churn label left out, as in production) ---
_db = sqlite3.connect(':memory:', check_same_thread=False)
_customers = pd.read_csv(CSV).drop(columns=['Churn'])
# blank TotalCharges are tenure=0 customers; numeric so SQL filters/sorts compare as numbers
_customers['TotalCharges'] = pd.to_numeric(_customers['TotalCharges'], errors='coerce').fillna(0.0)
_customers.to_sql('customers', _db, index=False)
_db.execute('PRAGMA query_only = ON')  # read-only enforced by SQLite, not by parsing the query

# --- knowledge base: one chunk per line under its section heading ---
_chunks, _section = [], ''
for line in (Path(__file__).parent / 'knowledge_base.md').read_text(encoding='utf-8').splitlines():
    if line.startswith('## '):
        _section = line[3:]
    elif line.strip() and not line.startswith(('#', '>')):
        _chunks.append(f'[{_section}] {line.strip()}')
_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(_chunks)
_kb = _vec.transform(_chunks)


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQLite SELECT on table `customers` (columns: customerID, gender, SeniorCitizen,
    Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges). Returns at most 20 rows."""
    try:
        cur = _db.execute(query)  # sqlite3 rejects multiple statements; query_only rejects writes
        rows = cur.fetchmany(20)
        cols = [d[0] for d in cur.description or []]
        return pd.DataFrame(rows, columns=cols).to_string(index=False) if cols else 'Query returned no rows.'
    except Exception as e:  # surface SQL errors so the model can fix its query
        return f'SQL error: {e}'


@tool
def churn_risk(customer_id: str) -> str:
    """Get the churn probability and risk level (low/medium/high) for one customer ID."""
    row = _customers[_customers['customerID'] == customer_id]
    if row.empty:
        return f'No customer with ID {customer_id}.'
    payload = json.loads(row.drop(columns=['customerID', 'gender']).iloc[0].to_json())  # native JSON types
    try:
        url = os.getenv('CHURN_API_URL')
        if url:
            r = httpx.post(f'{url.rstrip("/")}/predict', json=payload, timeout=60)
            r.raise_for_status()
            return json.dumps({**r.json(), 'source': 'azure-api'})
        from app import predict, Customer  # local fallback when no API URL is set
        return json.dumps({**predict(Customer(**payload)), 'source': 'local-model'})
    except Exception as e:  # keep the conversation alive if the API is down
        return f'churn_risk error: {e}'


@tool
def search_knowledge(query: str) -> str:
    """Search the care knowledge base (plans, retention offers, billing, tech support, contracts).
    Works with Arabic or English queries. Returns the 4 most relevant passages."""
    scores = cosine_similarity(_vec.transform([query]), _kb)[0]
    return '\n'.join(_chunks[i] for i in scores.argsort()[::-1][:4])


SYSTEM = """You are a customer-care assistant for a telecom operator.
- Reply in the customer's language: Egyptian Arabic if they write Arabic, English otherwise.
- Use tools for every fact: run_sql for account data, churn_risk before any retention offer,
  search_knowledge for plans, offers and policies. Never invent a price, offer or policy.
- When a customer may leave, check churn_risk, then pick the matching retention offer from the knowledge base.
- Keep answers short and end with the next step for the customer."""


def build_agent():
    llm = ChatGroq(model=os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'), temperature=0)
    return create_react_agent(llm, [run_sql, churn_risk, search_knowledge], prompt=SYSTEM)


def ask(agent, question: str) -> str:
    result = agent.invoke({'messages': [('user', question)]})
    for m in result['messages']:
        for call in getattr(m, 'tool_calls', None) or []:
            print(f'  -> {call["name"]}({json.dumps(call["args"], ensure_ascii=False)})')
    return result['messages'][-1].content


if __name__ == '__main__':
    agent = build_agent()
    if len(sys.argv) > 1:
        print(ask(agent, ' '.join(sys.argv[1:])))
    else:
        while (q := input('\nCustomer> ').strip()):
            print(ask(agent, q))
