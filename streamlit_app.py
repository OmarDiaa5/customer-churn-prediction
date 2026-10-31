"""Streamlit demo: churn scoring, retention what-if and bilingual care agent (Streamlit Community Cloud)."""
import html
import json
import os
import re
import time

import markdown
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
TOOL_CHIPS = {'run_sql': ('🗄️', 'Customer DB'), 'churn_risk': ('📈', 'Churn model'),
              'search_knowledge': ('📚', 'Knowledge base')}
CHAT_CSS = """
<style>
.st-key-chatshell {background: linear-gradient(180deg,#FFFFFF 0%,#FAFAFF 100%); border: 1px solid #E8E8F3;
  border-radius: 22px; box-shadow: 0 18px 50px -20px rgba(79,70,229,.25); overflow: hidden; gap: 0 !important;}
.chat-head {display:flex; align-items:center; gap:.85rem; padding: 1rem 1.3rem; border-bottom: 1px solid #EEF0F7;
  background: rgba(255,255,255,.85); backdrop-filter: blur(8px);}
.ava {width:42px; height:42px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  color:#fff; font-weight:700; font-size:1.05rem; flex-shrink:0;
  background: conic-gradient(from 210deg,#6366F1,#8B5CF6,#EC4899,#6366F1); box-shadow: 0 6px 16px -6px rgba(139,92,246,.7);}
.ava.sm {width:32px; height:32px; font-size:.85rem;}
.chat-head .name {font-weight:700; color:#111827; line-height:1.15;}
.chat-head .status {font-size:.8rem; color:#6B7280; display:flex; align-items:center; gap:.4rem;}
.dot-live {width:8px; height:8px; border-radius:50%; background:#22C55E; animation: pulse 2s infinite;}
.chat-head .model {margin-left:auto; font-size:.72rem; color:#6366F1; background:#EEF2FF; padding:.25rem .65rem;
  border-radius:999px; font-weight:600;}
@keyframes pulse {0%,100%{box-shadow:0 0 0 3px rgba(34,197,94,.22)} 50%{box-shadow:0 0 0 6px rgba(34,197,94,.04)}}
.st-key-chatbox {padding: .4rem 1.2rem 0 1.2rem;}
.row {display:flex; gap:.6rem; margin: .85rem 0; align-items:flex-end; animation: rise .35s ease-out;}
.row.user {justify-content:flex-end;}
@keyframes rise {from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none}}
.bubble {max-width: 76%; padding: .75rem 1rem; border-radius: 18px; line-height:1.6; font-size:.95rem;
  unicode-bidi: plaintext; text-align:start; overflow-wrap:anywhere;}
.bubble p {margin:.15rem 0;} .bubble ul, .bubble ol {margin:.3rem 1.1rem; padding:0;}
.bubble p, .bubble li {unicode-bidi: plaintext; text-align:start;}
.user .bubble {color:#fff; border-bottom-right-radius:6px;
  background: linear-gradient(135deg,#6366F1 0%,#8B5CF6 60%,#A855F7 100%); box-shadow:0 10px 24px -12px rgba(99,102,241,.8);}
.bot .bubble {color:#1F2937; background:#fff; border:1px solid #ECEEF5; border-bottom-left-radius:6px;
  box-shadow:0 8px 22px -16px rgba(17,24,39,.35);}
.bot .bubble strong {color:#4338CA;}
.bubble.rtl {direction: rtl; text-align: right;}
.bubble.rtl ul, .bubble.rtl ol {margin: .3rem 1.2rem .3rem 0;}
.chips {display:flex; flex-wrap:wrap; gap:.35rem; margin:-.35rem 0 .6rem 2.6rem;}
.chip {font-size:.72rem; color:#4B5563; background:#F3F4F8; border:1px solid #E5E7EF; padding:.18rem .6rem;
  border-radius:999px; cursor:help;}
.typing {display:inline-flex; gap:5px; padding:.95rem 1.1rem !important;}
.typing span {width:7px; height:7px; border-radius:50%; background:#A5B4FC; animation: blink 1.2s infinite;}
.typing span:nth-child(2){animation-delay:.2s} .typing span:nth-child(3){animation-delay:.4s}
@keyframes blink {0%,80%,100%{opacity:.3; transform:translateY(0)} 40%{opacity:1; transform:translateY(-4px)}}
.welcome {text-align:center; padding: 2.4rem 1rem 1.1rem;}
.welcome .ava {width:66px; height:66px; font-size:1.55rem; margin:0 auto .9rem;}
.welcome h3 {margin:.2rem 0; color:#111827;} .welcome p {color:#6B7280; margin:.3rem 0 0;}
.st-key-chips button {border-radius:999px !important; border:1px solid #E0E3F0 !important; background:#fff !important;
  color:#374151 !important; transition: all .15s;}
.st-key-chips button p {font-size:.83rem !important; unicode-bidi: plaintext;}
.st-key-chips button:hover {border-color:#8B5CF6 !important; color:#6D28D9 !important; transform: translateY(-1px);
  box-shadow:0 6px 14px -8px rgba(139,92,246,.6);}
.st-key-chatshell [data-testid="stChatInput"] {margin: .6rem 1.1rem 0 1.1rem; border-radius: 16px !important;
  border: 1px solid #E0E3F0 !important; box-shadow: 0 8px 24px -14px rgba(79,70,229,.45);}
.disclaimer {font-size:.72rem; color:#9CA3AF; text-align:center; padding:.45rem 1rem .8rem;}
</style>
"""
TYPING = ('<div class="row bot"><div class="ava sm">N</div><div class="bubble typing">'
          '<span></span><span></span><span></span></div></div>')


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum('\u0600' <= c <= '\u06FF' for c in letters) / len(letters) > 0.3


def bubble(role: str, text: str, trace=None) -> str:
    rtl = ' rtl' if is_arabic(text) else ''
    safe = html.escape(text)  # neither the model nor the user can inject HTML
    if role == 'user':
        return f'<div class="row user"><div class="bubble{rtl}">{safe.replace(chr(10), "<br>")}</div></div>'
    # LLMs often start a list right after a line of text; Markdown needs a blank line before it
    safe = re.sub(r'^((?!\s*(?:[-*]|\d+\.) ).+)\n(?=\s*(?:[-*]|\d+\.) )', r'\1\n\n', safe, flags=re.M)
    body = markdown.markdown(safe, extensions=['nl2br'])
    chips = ''
    if trace:
        chips = '<div class="chips">' + ''.join(
            '<span class="chip" title="{}">{} {}</span>'.format(
                html.escape(t['detail']), *TOOL_CHIPS.get(t['name'], ('🛠', t['name']))) for t in trace) + '</div>'
    return f'<div class="row bot"><div class="ava sm">N</div><div class="bubble{rtl}">{body}</div></div>{chips}'


with agent_tab:
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    if not os.getenv('GROQ_API_KEY'):
        st.warning('The agent needs a GROQ_API_KEY in the app secrets.')
        st.stop()

    @st.cache_resource
    def get_agent():
        return build_agent()

    from agent.care_agent import pick_model  # noqa: E402
    try:
        model = pick_model()
    except Exception as e:
        st.error(f'Could not reach Groq ({type(e).__name__}): {getattr(e, "message", "")[:300]}')
        st.stop()

    st.session_state.setdefault('history', [])
    clicked = None
    with st.container(key='chatshell'):
        st.markdown(
            '<div class="chat-head"><div class="ava">N</div><div><div class="name">Nour · Care Assistant</div>'
            '<div class="status"><span class="dot-live"></span>Online · replies in Arabic &amp; English</div></div>'
            f'<div class="model">{html.escape(model)}</div></div>', unsafe_allow_html=True)
        box = st.container(height=500, border=False, key='chatbox')
        with box:
            if not st.session_state.history:
                st.markdown("<div class='welcome'><div class='ava'>N</div><h3>Hi, I'm Nour 👋</h3>"
                            '<p>Ask about an account, churn risk, plans or billing.</p>'
                            '<p>اسألني بالعربي أو بالإنجليزي</p></div>', unsafe_allow_html=True)
                with st.container(key='chips'):
                    cols = st.columns(2)
                    for i, q in enumerate(EXAMPLES):
                        if cols[i % 2].button(q, key=f'ex{i}', use_container_width=True):
                            clicked = q
            for role, text, trace in st.session_state.history:
                st.markdown(bubble(role, text, trace), unsafe_allow_html=True)
        question = st.chat_input('Message Nour · اكتب رسالتك') or clicked
        st.markdown('<div class="disclaimer">Answers come from read-only SQL, the churn model and a sample '
                    "knowledge base. Plans and offers are demo content, not a real operator's.</div>",
                    unsafe_allow_html=True)

    if question:  # store the message and rerun so it (not the welcome screen) shows while the agent works
        st.session_state.history.append(('user', question, []))
        st.session_state.pending = True
        st.rerun()

    if st.session_state.pop('pending', False):
        msgs = [(r, t) for r, t, _ in st.session_state.history]
        with box:
            slot = st.empty()
            slot.markdown(TYPING, unsafe_allow_html=True)
            try:
                out = get_agent().invoke({'messages': msgs})
            except Exception as e:
                name = type(e).__name__
                if 'Authentication' in name:
                    err = 'The Groq API key was rejected. Update GROQ_API_KEY in the app secrets.'
                elif 'RateLimit' in name:
                    err = 'Groq rate limit reached. Try again in a minute.'
                else:  # Groq error bodies carry no secrets, so show them to make failures diagnosable
                    err = f'The agent hit an error ({name}): {getattr(e, "message", "")[:300]}'
                slot.empty()
                st.session_state.history.pop()
                st.error(err)
                st.stop()
            new = out['messages'][len(msgs):]
            trace = [{'name': c['name'], 'detail': f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"}
                     for m_ in new for c in (getattr(m_, 'tool_calls', None) or [])]
            answer = out['messages'][-1].content
            shown = []
            for w in answer.split(' '):  # type the reply out like a live chat
                shown.append(w)
                slot.markdown(bubble('assistant', ' '.join(shown)), unsafe_allow_html=True)
                time.sleep(0.015)
        st.session_state.history.append(('assistant', answer, trace))
        st.rerun()

    if st.session_state.history and st.button('↺  New conversation', type='tertiary'):
        st.session_state.history = []
        st.rerun()
