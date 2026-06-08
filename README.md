# Project 1 — Website Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that crawls a website, indexes its content, and answers user questions using a local LLM served via LM Studio.

## Architecture

```
Website URL
    ↓
[Crawler]  →  Raw text pages
    ↓
[Chunker + Embedder]  →  Vector Store (ChromaDB, local)
    ↓
[Streamlit UI]
    ↓ (user question)
[Retriever]  →  top-k relevant chunks
    ↓
[LM Studio LLM]  →  Answer
    ↓
[Streamlit UI]  →  Display answer
```

## Requirements

- Python 3.10+
- LM Studio running locally with **Qwen2.5-3B-Instruct** loaded
  - Server must be enabled: `LM Studio → Local Server → Start Server`
  - Default URL: `http://localhost:1234/v1`

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/website-chatbot-rag.git
cd website-chatbot-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start LM Studio local server
#    - Open LM Studio
#    - Load Qwen2.5-3B-Instruct model
#    - Go to Local Server tab → Start Server

# 5. Run the app
streamlit run app.py
```

## Usage

1. Enter the website URL you want to crawl
2. Click **"Crawl & Index"** — the system will crawl and embed the content
3. Ask questions about the website content in the chat box

## Project Structure

```
project1/
├── app.py                  # Streamlit UI entry point
├── src/
│   ├── crawler.py          # Web crawler
│   ├── embedder.py         # Text chunking + embedding
│   ├── retriever.py        # Vector store + retrieval
│   └── llm_client.py       # LM Studio API client
├── data/                   # Local ChromaDB storage (auto-created)
├── tests/
│   └── test_crawler.py
├── requirements.txt
└── README.md
```

## Notes

- Crawling is limited to the same domain (no external links followed)
- Max crawl depth: 3 levels
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings (fast, local, no API key)
- ChromaDB stores vectors locally in `./data/chroma_db`
