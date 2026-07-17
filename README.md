# ProseVector — AI Book Recommendation Chatbot

RAG chatbot with custom HTML frontend. Recommends books from semantic search.

## Setup (do this once)

### 1 — Install dependencies
    pip install -r requirements.txt

### 2 — Install Ollama and pull the LLM
    # Download Ollama: https://ollama.com
    ollama pull phi3          # 2.3 GB — use if laptop has 4-6 GB RAM
    # or
    ollama pull mistral       # 4.1 GB — use if laptop has 8+ GB RAM

### 3 — Build the FAISS book index (run once, ~5 minutes)
    python ingest.py

## Run the app (every time)

### Terminal 1 — start Ollama
    ollama serve

### Terminal 2 — start the web server
    python app.py

### Then open in browser
    http://localhost:7860

## Tech Stack
- Flask — backend server and API
- LangChain — RAG pipeline orchestration
- HuggingFace Datasets — CMU Book Summaries (5,000 books)
- sentence-transformers/all-MiniLM-L6-v2 — local embedding model (free)
- FAISS — local vector database (no server needed)
- Ollama phi3/mistral — local LLM (completely free, works offline)
- Vanilla HTML/CSS/JS — custom frontend (no React, no framework)
