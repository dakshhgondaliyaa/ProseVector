import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

FAISS_INDEX_PATH     = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL         = "phi3"      # change to "mistral" if you have 8GB+ RAM
TOP_K_RESULTS        = 3

app = Flask(__name__, static_folder="frontend")
CORS(app)

# ── Load FAISS index ──────────────────────────────────────────────────────────
def load_vectorstore():
    print("Loading FAISS index from disk...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("FAISS index loaded.")
    return vectorstore

# ── Build RAG chain ───────────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    print(f"Connecting to Ollama model: {OLLAMA_MODEL}")
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    prompt_template = """You are ProseVector, a helpful and expert AI book librarian.
A reader has described a book they are looking for. Use the book summaries provided in the context below to recommend the best matching book.

Your response must:
1. State the exact book title in double quotes like "Dune" and the author name.
2. In one or two sentences, explain why the plot from the context matches the reader's description.
3. Be warm and conversational.

IMPORTANT — Before recommending a book, check: does the context actually contain a book whose plot, themes, or subject matter genuinely matches what the reader described? If YES, recommend it. If NO — if the retrieved summaries are about completely different topics, genres, or subjects — you MUST respond with exactly:
"I couldn't find an exact match. Could you describe the plot in more detail?"

Do NOT guess or force a loose connection. Only recommend a book if there is a clear, genuine match between the reader's description and a book summary in the context.

Example of correct fallback:
- Reader asks: "A book about training dolphins to deliver mail underwater"
- Context contains summaries about war novels and romance fiction
- Correct response: "I couldn't find an exact match. Could you describe the plot in more detail?"

Context:
{context}

Reader's description: {input}

Recommendation:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"],
    )

    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("RAG chain ready.")
    return chain

# ── Routes ────────────────────────────────────────────────────────────────────

# Serve the frontend HTML
@app.route("/")
def serve_index():
    return send_from_directory("frontend", "index.html")

# Chat API endpoint — called by the frontend via fetch()
@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Empty message"}), 400
        
    # Combine the previous user message with the current one to give context
    search_query = message
    if history:
        last_user_msg = ""
        for msg in reversed(history):
            if msg["role"] == "user":
                last_user_msg = msg["text"]
                break
        if last_user_msg:
            search_query = f"Previous thought: {last_user_msg}\n\nFollow-up: {message}"

    try:
        result  = rag_chain.invoke({"input": search_query})
        answer  = result["answer"]
        sources = [
            doc.page_content[:120]
            for doc in result.get("context", [])
        ]
        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(FAISS_INDEX_PATH):
        print("ERROR: FAISS index not found. Run: python ingest.py")
        exit(1)

    print("=" * 50)
    print("  ProseVector — AI Book Recommender")
    print("=" * 50)

    vectorstore = load_vectorstore()
    rag_chain   = build_rag_chain(vectorstore)

    print("\nOpen your browser at: http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
