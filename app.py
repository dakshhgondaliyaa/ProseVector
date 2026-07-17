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
TOP_K_RESULTS        = 5

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

    prompt_template = """You are ProseVector, an expert AI book librarian.
A reader has described a book they are looking for. Use ONLY the book information provided in the context below to recommend the best matching book. DO NOT invent or hallucinate any books, authors, or plots.

Your response must:
1. State the exact book title in double quotes like "Dune" and the author name.
2. In one sentence, explain why the plot from the context matches the reader's description.
3. Be warm and conversational.

If none of the books in the context match the reader's description, you must reply EXACTLY with this sentence and nothing else:
"I couldn't find an exact match. Could you describe the plot in more detail?"

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

    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        result  = rag_chain.invoke({"input": message})
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
