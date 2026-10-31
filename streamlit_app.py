"""Streamlit demo: churn scoring + bilingual customer-care agent (deployed on Streamlit Community Cloud)."""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title='Telco Churn & Care Agent', page_icon='📉', layout='centered')

# Groq key comes from Streamlit secrets in the cloud, or the environment locally
try:
    os.environ.setdefault('GROQ_API_KEY', st.secrets['GROQ_API_KEY'])
except Exception:  # no secrets.toml locally, or key not set
    pass

from agent.care_agent import _customers, build_agent, churn_risk  # noqa: E402  (after env is set)
from app import BUNDLE  # noqa: E402

st.title('Telco Churn Risk & Care Agent')
st.caption('XGBoost + SMOTE churn model on the IBM Telco dataset (7,043 customers) · '
           f"held-out recall {BUNDLE['metrics']['recall']}, ROC AUC {BUNDLE['metrics']['roc_auc']}")

score_tab, agent_tab = st.tabs(['Churn scoring', 'Care agent (عربي / English)'])

with score_tab:
    cid = st.selectbox('Customer ID', _customers['customerID'], index=0)
    row = _customers[_customers['customerID'] == cid].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric('Tenure (months)', int(row['tenure']))
    c2.metric('Monthly charges', f"${row['MonthlyCharges']:.2f}")
    c3.metric('Contract', row['Contract'])
    result = json.loads(churn_risk.invoke({'customer_id': cid}))
    st.subheader(f"Risk: {result['risk'].upper()}")
    st.progress(result['churn_score'], text=f"Churn score {result['churn_score']:.3f}")
    st.caption('The score is recall-weighted, not a calibrated probability. Test churn rate by band: '
               'high 53%, medium 19%, low 4% (base rate 26.5%).')

with agent_tab:
    st.caption('Ask as a customer or a care rep, e.g. "Customer 7590-VHVEG wants to cancel, what can we offer?" '
               'or «عميل رقمه 7590-VHVEG عايز يلغي الخط، نعرض عليه إيه؟». '
               'Plans and offers are sample demo content, not a real operator\'s.')
    if not os.getenv('GROQ_API_KEY'):
        st.warning('GROQ_API_KEY is not configured.')
        st.stop()

    @st.cache_resource
    def get_agent():
        return build_agent()

    st.session_state.setdefault('history', [])
    for role, text, trace in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)
            if trace:
                with st.expander('Tool trace'):
                    st.code('\n'.join(trace), language='text')

    if question := st.chat_input('Type a message / اكتب رسالتك'):
        with st.chat_message('user'):
            st.markdown(question)
        msgs = [(r, t) for r, t, _ in st.session_state.history] + [('user', question)]
        with st.chat_message('assistant'), st.spinner('Thinking...'):
            out = get_agent().invoke({'messages': msgs})
            new = out['messages'][len(msgs):]
            trace = [f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                     for m in new for c in (getattr(m, 'tool_calls', None) or [])]
            answer = out['messages'][-1].content
            st.markdown(answer)
            if trace:
                with st.expander('Tool trace'):
                    st.code('\n'.join(trace), language='text')
        st.session_state.history += [('user', question, []), ('assistant', answer, trace)]
