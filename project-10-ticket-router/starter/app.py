"""Streamlit starter app — Project 10 (Helpdesk Ticket Router).

This app is infrastructure only. It loads the synthetic ticket data,
shows the label distribution, and gives you a text box to type a ticket.
Clicking "Classify" calls the (empty) student core and shows the
resulting `NotImplementedError` until you implement `train_model` and
`predict` in `starter/student_core.py`. This page never searches, infers,
or calls a hosted model on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run starter/app.py` to import the sibling `starter`
# package regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter import student_core  # noqa: E402
from starter.tokenizer import tokenize  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="IT Helpdesk Ticket Router — starter", layout="wide")
st.title("IT Helpdesk Ticket Router — starter")
st.markdown(
    "The **classifier is student work**. This page only loads data and "
    "shows a text box: it must not train, predict, or call a hosted "
    "model on its own."
)


@st.cache_data
def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticket_id", "text", "label"])
    return pd.read_csv(path)


train_df = load_split("train")

with st.sidebar:
    st.header("Dataset")
    split_name = st.selectbox("Split", ["train", "validation", "test", "hard_cases"])
    st.caption("Regenerate with `python3.11 scripts/generate_tickets.py --seed 42`.")

split_df = load_split(split_name)

col_data, col_classify = st.columns([2, 1])

with col_data:
    st.subheader(f"{split_name}.csv — {len(split_df)} rows")
    if not split_df.empty:
        st.bar_chart(split_df["label"].value_counts())
        st.dataframe(split_df.sample(min(10, len(split_df)), random_state=0), use_container_width=True)
    else:
        st.warning(f"No data found. Run the generator first: `python3.11 scripts/generate_tickets.py`.")

with col_classify:
    st.subheader("Try a ticket")
    default_text = "I cannot log into my account and the password reset email never arrives."
    ticket_text = st.text_area("Ticket text", value=default_text, height=140)
    st.caption(f"Tokens (optional helper preview): {tokenize(ticket_text)[:12]}")

    if st.button("Classify"):
        try:
            model = student_core.train_model(
                list(train_df["text"]), list(train_df["label"])
            ) if not train_df.empty else None
            predictions = student_core.predict(model, [ticket_text])
            st.success(f"Predicted label: {predictions[0]}")
        except NotImplementedError as exc:
            st.error(f"Not implemented yet: {exc}")

st.subheader("Label distribution across splits")
rows = []
for name in ["train", "validation", "test"]:
    df = load_split(name)
    if df.empty:
        continue
    counts = df["label"].value_counts()
    for label, count in counts.items():
        rows.append({"split": name, "label": label, "count": int(count)})
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
