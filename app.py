"""
app.py
------
Streamlit UI for the Website RAG Chatbot.
Run with: streamlit run app.py
"""

import logging
import sys
import os

import streamlit as st

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))
from src.crawler import crawl
from src.embedder import Embedder
from src.retriever import Retriever
from src.llm_client import LLMClient

logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Website Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Website RAG Chatbot")
st.caption("Powered by LM Studio (Qwen2.5-3B-Instruct) + ChromaDB")

# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": str, "content": str}
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "crawled_url" not in st.session_state:
    st.session_state.crawled_url = ""

# ──────────────────────────────────────────────
# Sidebar — Crawl & Index
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # LM Studio status
    llm = LLMClient()
    if llm.is_available():
        st.success("✅ LM Studio is running")
    else:
        st.error(
            "❌ LM Studio not detected at http://localhost:1234\n\n"
            "Please open LM Studio, load the model, and start the Local Server."
        )

    st.divider()

    st.subheader("🌐 Website Crawl")
    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        help="Enter the root URL of the website you want to index.",
    )

    col1, col2 = st.columns(2)
    with col1:
        max_pages = st.number_input("Max pages", min_value=1, max_value=200, value=30)
    with col2:
        max_depth = st.number_input("Max depth", min_value=1, max_value=5, value=3)

    reset_index = st.checkbox("Reset existing index before crawling", value=True)

    crawl_button = st.button("🔍 Crawl & Index", use_container_width=True, type="primary")

    if crawl_button:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            with st.spinner("Crawling website..."):
                try:
                    pages = crawl(
                        start_url=url_input.strip(),
                        max_depth=int(max_depth),
                        max_pages=int(max_pages),
                    )
                except Exception as e:
                    st.error(f"Crawl failed: {e}")
                    pages = []

            if pages:
                st.info(f"Crawled {len(pages)} pages. Indexing...")
                progress_bar = st.progress(0)

                def update_progress(current, total):
                    progress_bar.progress(current / total)

                embedder = Embedder()
                if reset_index:
                    embedder.reset()

                try:
                    n_chunks = embedder.index_pages(pages, progress_callback=update_progress)
                    progress_bar.progress(1.0)
                    st.success(f"✅ Indexed {n_chunks} chunks from {len(pages)} pages.")
                    st.session_state.indexed = True
                    st.session_state.crawled_url = url_input.strip()
                    st.session_state.chat_history = []  # Reset chat on new index
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
            else:
                st.warning("No pages were crawled. Check the URL and try again.")

    # Show current index status
    st.divider()
    try:
        retriever_check = Retriever()
        count = retriever_check.count
        if count > 0:
            st.info(f"📚 Index has **{count}** chunks stored")
            if st.session_state.crawled_url:
                st.caption(f"Source: {st.session_state.crawled_url}")
            st.session_state.indexed = True
        else:
            st.info("Index is empty. Crawl a website to get started.")
    except Exception:
        pass

    st.divider()
    st.subheader("🔧 LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    top_k = st.slider("Retrieved chunks (top-k)", 1, 10, 5)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ──────────────────────────────────────────────
# Main — Chat Interface
# ──────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("👈 Enter a website URL in the sidebar and click **Crawl & Index** to get started.")
    st.stop()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask something about the website..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Retrieve relevant chunks
    retriever = Retriever()
    chunks = retriever.retrieve(question, top_k=int(top_k))

    if not chunks:
        answer = "I couldn't find any relevant information in the indexed content."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        context = retriever.build_context(chunks)

        # Stream the answer
        llm_client = LLMClient()
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for token in llm_client.stream_answer(
                    question=question,
                    context=context,
                    chat_history=[
                        m for m in st.session_state.chat_history[:-1]  # exclude latest user msg
                    ],
                    temperature=temperature,
                ):
                    full_answer += token
                    answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)

                # Show sources in an expander
                with st.expander("📎 Sources used"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(
                            f"**[{i}] [{chunk['title']}]({chunk['url']})** "
                            f"— similarity: `{chunk['score']}`"
                        )
                        st.caption(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                        st.divider()

            except ConnectionError as e:
                full_answer = f"⚠️ {e}"
                answer_placeholder.error(full_answer)

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
