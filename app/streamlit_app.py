"""
Streamlit UI for the FastAPI Docs Assistant.

Two tabs:
1. Chat -- ask a question, get a grounded answer + sources, give feedback.
2. Monitoring -- dashboard built from the interaction log (db.py).

Run:
    streamlit run app/streamlit_app.py
"""

import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from app.db import fetch_all, log_interaction, record_feedback
from app.rag import answer_question
from app.search import get_index

st.set_page_config(page_title="FastAPI Docs Assistant", page_icon="⚡", layout="wide")

if "index" not in st.session_state:
    with st.spinner("Loading search index..."):
        st.session_state.index = get_index()

if "last_interaction_id" not in st.session_state:
    st.session_state.last_interaction_id = None

tab_chat, tab_monitor = st.tabs(["💬 Chat", "📊 Monitoring"])

# ----------------------------------------------------------------- Chat tab
with tab_chat:
    st.title("⚡ FastAPI Docs Assistant")
    st.caption(
        "Ask questions about FastAPI. Answers are grounded in the official "
        "documentation via hybrid (keyword + vector) retrieval."
    )

    with st.sidebar:
        st.header("Settings")
        method = st.selectbox("Retrieval method", ["hybrid", "keyword", "vector"], index=0)
        top_k = st.slider("Chunks to retrieve", 1, 10, 5)
        use_rewriting = st.checkbox("Query rewriting (bonus)", value=False,
                                     help="Expand the question into alternative phrasings and fuse retrieval across all of them. Needs an LLM key.")
        use_reranking = st.checkbox("LLM reranking (bonus)", value=False,
                                     help="Retrieve a larger candidate set, then have the LLM re-score and reorder before answering. Needs an LLM key.")
        st.divider()
        st.caption(
            "No LLM_PROVIDER / API key configured shows demo-mode answers "
            "(top retrieved passage instead of a generated answer, bonus "
            "features no-op). Set OPENAI_API_KEY or ANTHROPIC_API_KEY to "
            "enable real generation."
        )

    question = st.text_input("Your question", placeholder="How do I add a background task?")
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        start = time.time()
        with st.spinner("Retrieving context and generating an answer..."):
            result = answer_question(
                question, index=st.session_state.index, top_k=top_k, method=method,
                use_query_rewriting=use_rewriting, use_reranking=use_reranking,
            )
        elapsed_ms = (time.time() - start) * 1000

        interaction_id = log_interaction(
            question=question,
            answer=result["answer"],
            retrieval_method=method,
            sources_json=json.dumps(result["sources"]),
            response_time_ms=elapsed_ms,
        )
        st.session_state.last_interaction_id = interaction_id
        st.session_state.last_result = result
        st.session_state.last_elapsed_ms = elapsed_ms

    if st.session_state.get("last_result"):
        result = st.session_state.last_result
        st.markdown("### Answer")
        st.write(result["answer"])
        st.caption(f"Responded in {st.session_state.last_elapsed_ms:.0f} ms "
                   f"using **{result['retrieval_method']}** retrieval")

        with st.expander("Sources used"):
            for s in result["sources"]:
                st.markdown(f"- **{s['source']}** — *{s['heading']}*  \n  {s['url']}")

        col1, col2, _ = st.columns([1, 1, 6])
        with col1:
            if st.button("👍 Helpful"):
                record_feedback(st.session_state.last_interaction_id, 1)
                st.success("Thanks for the feedback!")
        with col2:
            if st.button("👎 Not helpful"):
                record_feedback(st.session_state.last_interaction_id, -1)
                st.info("Thanks -- noted for monitoring.")

# ------------------------------------------------------------ Monitoring tab
with tab_monitor:
    st.title("📊 Monitoring Dashboard")
    rows = fetch_all()

    if not rows:
        st.info("No interactions logged yet -- ask a question in the Chat tab first.")
    else:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["date"] = df["timestamp"].dt.date

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total questions", len(df))
        c2.metric("Avg response time (ms)", f"{df['response_time_ms'].mean():.0f}")
        thumbs_up = (df["feedback"] == 1).sum()
        thumbs_down = (df["feedback"] == -1).sum()
        rated = thumbs_up + thumbs_down
        c3.metric("Feedback rate", f"{(rated / len(df) * 100):.0f}%")
        c4.metric("Satisfaction (of rated)", f"{(thumbs_up / rated * 100):.0f}%" if rated else "n/a")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Questions per day")
            daily = df.groupby("date").size()
            st.bar_chart(daily)

            st.subheader("Retrieval method usage")
            st.bar_chart(df["retrieval_method"].value_counts())

            st.subheader("Response time distribution (ms)")
            st.bar_chart(df["response_time_ms"].round(-2).value_counts().sort_index())

        with col_b:
            st.subheader("Feedback breakdown")
            fb_counts = df["feedback"].map(
                {1: "👍 Helpful", -1: "👎 Not helpful"}
            ).fillna("No feedback").value_counts()
            st.bar_chart(fb_counts)

            st.subheader("Response time over time")
            st.line_chart(df.set_index("timestamp")["response_time_ms"])

            st.subheader("Recent questions")
            st.dataframe(
                df[["timestamp", "question", "retrieval_method", "feedback"]].head(10),
                width="stretch",
            )
