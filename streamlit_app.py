"""Streamlit demo: churn scoring, retention what-if and bilingual care agent (Streamlit Community Cloud)."""
import html
import json
import os

import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title='Telco Churn & Care Agent', page_icon='📶', layout='wide')

# Groq key comes from Streamlit secrets in the cloud, or the environment locally
try:
    os.environ.setdefault('GROQ_API_KEY', st.secrets['GROQ_API_KEY'].strip())
except Exception:  # no secrets.toml locally, or key not set
    pass

from agent.care_agent import _customers, build_agent  # noqa: E402  (after env is set)
from app import BUNDLE, Customer, predict  # noqa: E402

RISK_STYLE = {'high': ('#DC2626', '#FEE2E2', 'High risk'), 'medium': ('#D97706', '#FEF3C7', 'Medium risk'),
              'low': ('#059669', '#D1FAE5', 'Low risk')}
FEATURE_LABELS = {
    'tenure': 'Tenure', 'MonthlyCharges': 'Monthly charges', 'TotalCharges': 'Total charges',
    'Contract': 'Contract length', 'SeniorCitizen': 'Senior citizen', 'Partner': 'Has partner',
    'Dependents': 'Has dependents', 'PaperlessBilling': 'Paperless billing', 'TechSupport': 'Tech support',
    'OnlineSecurity': 'Online security', 'OnlineBackup': 'Online backup', 'DeviceProtection': 'Device protection',
    'StreamingTV': 'Streaming TV', 'StreamingMovies': 'Streaming movies', 'PhoneService': 'Phone service',
    'SpendingTier': 'Spending tier', 'TenureGroup': 'Tenure group', 'Cluster': 'Customer segment',
    'InternetService_Fiber optic': 'Fiber internet', 'InternetService_No': 'No internet',
    'PaymentMethod_Electronic check': 'Pays by electronic check', 'PaymentMethod_Mailed check': 'Pays by mailed check',
    'PaymentMethod_Credit card (automatic)': 'Auto card payment', 'MultipleLines_Yes': 'Multiple lines',
    'MultipleLines_No phone service': 'No phone service',
}
OFFERS = {
    'Move to a 1-year contract': {'Contract': 'One year'},
    'Switch to automatic card payment': {'PaymentMethod': 'Credit card (automatic)'},
    'Add free tech support': {'TechSupport': 'Yes'},
}
EXAMPLES = [
    'Customer 7590-VHVEG wants to cancel. What can we offer?',
    'عميل رقمه 7590-VHVEG عايز يلغي الخط، نعرض عليه إيه؟',
    'How many month-to-month customers have fiber internet?',
    'النت عندي بطيء أعمل إيه؟',
]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero h1 {font-size: 2rem; margin-bottom: .2rem;}
.hero p {color: #6B7280; margin-top: 0;}
.card {background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 1.1rem 1.3rem; height: 100%;}
.card h4 {margin: 0 0 .6rem 0; font-size: .85rem; color: #6B7280; text-transform: uppercase; letter-spacing: .04em;}
.risk {border-radius: 14px; padding: 1.2rem 1.4rem;}
.risk .label {font-size: 1.6rem; font-weight: 700;}
.risk .score {font-size: .95rem; opacity: .85;}
.kv {display: flex; justify-content: space-between; padding: .28rem 0; border-bottom: 1px dashed #F0F1F5;}
.kv span:first-child {color: #6B7280;}
.driver {display: flex; justify-content: space-between; padding: .3rem 0;}
.up {color: #DC2626; font-weight: 600;} .down {color: #059669; font-weight: 600;}
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li
  {unicode-bidi: plaintext; text-align: start;}
.trace code {font-size: .78rem;}
</style>
""", unsafe_allow_html=True)


def score(payload: dict) -> dict:
    return predict(Customer(**payload))


def drivers(payload: dict, k: int = 4):
    """Top features pushing this customer's score up or down (XGBoost SHAP contributions)."""
    X = BUNDLE['featurizer'].transform(pd.DataFrame([payload]))[BUNDLE['columns']]
    contrib = BUNDLE['model'].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    s = pd.Series(contrib, index=BUNDLE['columns'])
    return s.reindex(s.abs().sort_values(ascending=False).index).head(k)


RAW_LABELS = {'InternetService': 'Internet', 'PaymentMethod': 'Payment', 'MultipleLines': 'Multiple lines'}


def describe(feature: str, p: dict) -> str:
    """Show the customer's actual value, so a one-hot column reads 'Internet: DSL', not 'Fiber internet'."""
    base = feature.split('_', 1)[0]
    if base in RAW_LABELS:
        return f"{RAW_LABELS[base]}: {p[base]}"
    if base == 'tenure':
        return f"Tenure: {p['tenure']} month{'s' if p['tenure'] != 1 else ''}"
    if base in p and base not in ('MonthlyCharges', 'TotalCharges'):
        value = {0: 'No', 1: 'Yes'}.get(p[base], p[base]) if base == 'SeniorCitizen' else p[base]
        return f"{FEATURE_LABELS.get(base, base)}: {value}"
    if base in ('MonthlyCharges', 'TotalCharges'):
        return f"{FEATURE_LABELS[base]}: ${p[base]:,.2f}"
    return FEATURE_LABELS.get(feature, feature)


def payload_for(cid: str) -> dict:
    row = _customers[_customers['customerID'] == cid]
    return json.loads(row.drop(columns=['customerID', 'gender']).iloc[0].to_json())


# ---------- header ----------
m = BUNDLE['metrics']
st.markdown('<div class="hero"><h1>📶 Telco Churn Risk & Care Agent</h1>'
            '<p>XGBoost + SMOTE on the IBM Telco dataset (7,043 customers), served with a bilingual '
            'LangGraph care agent.</p></div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric('Held-out recall', f"{m['recall']:.2f}")
c2.metric('ROC AUC', f"{m['roc_auc']:.2f}")
c3.metric('Customers', f"{len(_customers):,}")
c4.metric('Base churn rate', '26.5%')

score_tab, agent_tab = st.tabs(['🔎  Churn scoring', '💬  Care agent  ·  عربي / English'])

# ---------- scoring ----------
with score_tab:
    cid = st.selectbox('Customer', _customers['customerID'], index=0,
                       help='Type to search any of the 7,043 customer IDs.')
    p = payload_for(cid)
    res = score(p)
    color, bg, label = RISK_STYLE[res['risk']]

    left, mid, right = st.columns([1.05, 1, 1.15])
    with left:
        st.markdown(
            f'<div class="risk" style="background:{bg};color:{color}">'
            f'<div class="label">{label}</div>'
            f'<div class="score">Churn score {res["churn_score"]:.2f} · segment {res["kmeans_segment"]}</div></div>',
            unsafe_allow_html=True)
        st.progress(res['churn_score'])
        st.caption('Recall-weighted score, not a calibrated probability. Actual test churn by band: '
                   'high 53% · medium 19% · low 4%.')
    with mid:
        rows = [('Tenure', f"{p['tenure']} month{'s' if p['tenure'] != 1 else ''}"), ('Contract', p['Contract']),
                ('Internet', p['InternetService']), ('Payment', p['PaymentMethod']),
                ('Monthly bill', f"${p['MonthlyCharges']:.2f}"), ('Tech support', p['TechSupport'])]
        st.markdown('<div class="card"><h4>Profile</h4>' + ''.join(
            f'<div class="kv"><span>{k}</span><span>{v}</span></div>' for k, v in rows) + '</div>',
            unsafe_allow_html=True)
    with right:
        d = drivers(p)
        st.markdown('<div class="card"><h4>Why this score</h4>' + ''.join(
            f'<div class="driver"><span>{describe(f, p)}</span>'
            f'<span class="{"up" if v > 0 else "down"}">{"▲ raises" if v > 0 else "▼ lowers"} risk</span></div>'
            for f, v in d.items()) + '</div>', unsafe_allow_html=True)

    st.markdown('#### Retention what-if')
    available = [o for o, change in OFFERS.items()
                 if all(p[k] != v for k, v in change.items())
                 and not (o == 'Move to a 1-year contract' and p['Contract'] != 'Month-to-month')
                 and not (o == 'Add free tech support' and p['InternetService'] == 'No')]
    chosen = st.multiselect('Simulate an offer', available, default=[],
                            placeholder='Pick one or more offers to see the new score')
    if chosen:
        changed = dict(p)
        for o in chosen:
            changed.update(OFFERS[o])
        new = score(changed)
        delta = new['churn_score'] - res['churn_score']
        a, b = st.columns(2)
        a.metric('Score after offer', f"{new['churn_score']:.2f}", f"{delta:+.2f}", delta_color='inverse')
        b.metric('Risk band after offer', RISK_STYLE[new['risk']][2])

# ---------- agent ----------
with agent_tab:
    st.caption('Answers come from tools: read-only SQL over the customer table, the churn model, and a '
               'knowledge base. Plans and offers are sample demo content, not a real operator\'s.')
    if not os.getenv('GROQ_API_KEY'):
        st.warning('The agent needs a GROQ_API_KEY in the app secrets.')
        st.stop()

    @st.cache_resource
    def get_agent():
        return build_agent()

    from agent.care_agent import pick_model  # noqa: E402
    try:
        st.caption(f'Model: `{pick_model()}` on Groq')
    except Exception as e:
        st.error(f'Could not reach Groq ({type(e).__name__}): {getattr(e, "message", "")[:300]}')
        st.stop()

    st.session_state.setdefault('history', [])
    cols = st.columns(len(EXAMPLES))
    clicked = next((q for col, q in zip(cols, EXAMPLES) if col.button(q, use_container_width=True)), None)

    for role, text, trace in st.session_state.history:
        with st.chat_message(role, avatar='🧑' if role == 'user' else '🤖'):
            st.markdown(text)
            if trace:
                with st.expander(f'🛠 {len(trace)} tool call(s)'):
                    st.markdown('<div class="trace">' + '<br>'.join(f'<code>{html.escape(t)}</code>' for t in trace)
                                + '</div>', unsafe_allow_html=True)

    question = st.chat_input('Type a message · اكتب رسالتك') or clicked
    if question:
        with st.chat_message('user', avatar='🧑'):
            st.markdown(question)
        msgs = [(r, t) for r, t, _ in st.session_state.history] + [('user', question)]
        with st.chat_message('assistant', avatar='🤖'):
            try:
                with st.spinner('Checking account, risk and policies...'):
                    out = get_agent().invoke({'messages': msgs})
            except Exception as e:
                name = type(e).__name__
                if 'Authentication' in name:
                    st.error('The Groq API key was rejected. Update GROQ_API_KEY in the app secrets.')
                elif 'RateLimit' in name:
                    st.error('Groq rate limit reached. Try again in a minute.')
                else:
                    # Groq error bodies carry no secrets, so show them to make failures diagnosable
                    st.error(f'The agent hit an error ({name}): {getattr(e, "message", "")[:300]}')
                st.stop()
            new = out['messages'][len(msgs):]
            trace = [f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                     for msg in new for c in (getattr(msg, 'tool_calls', None) or [])]
            answer = out['messages'][-1].content
            st.markdown(answer)
            if trace:
                with st.expander(f'🛠 {len(trace)} tool call(s)'):
                    st.markdown('<div class="trace">' + '<br>'.join(f'<code>{html.escape(t)}</code>' for t in trace)
                                + '</div>', unsafe_allow_html=True)
        st.session_state.history += [('user', question, []), ('assistant', answer, trace)]
