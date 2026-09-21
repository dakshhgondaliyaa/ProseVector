import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

FAISS_INDEX_PATH     = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL         = "phi3"      # change to "mistral" if you have 8GB+ RAM
TOP_K_RESULTS        = 5

app = Flask(__name__, static_folder="frontend")
CORS(app)

# ── Database Initialization ───────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ── Load FAISS index ──────────────────────────────────────────────────────────
def load_vectorstore():
    print("Loading FAISS index from disk...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
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
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # 1. History aware retriever
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answer chain
    qa_system_prompt = """You are ProseVector, a helpful and expert AI book librarian.
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
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    print("RAG chain ready.")
    return chain

# ── Routes ────────────────────────────────────────────────────────────────────

# Serve the login HTML at root
@app.route("/")
def serve_login():
    return send_from_directory("frontend", "login.html")

# Serve the main chat interface
@app.route("/app")
def serve_index():
    return send_from_directory("frontend", "index.html")

# ── Authentication API ────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")
    
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required"}), 400
        
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return jsonify({"success": False, "error": "Username already exists"}), 409
            
        password_hash = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()
        
    return jsonify({"success": True})

@app.route("/api/login", methods=["POST"])
def login_api():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")
    
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required"}), 400
        
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        
        if row and check_password_hash(row[0], password):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()

# ── Chat API ──────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Empty message"}), 400
        
    # Convert frontend history to LangChain messages
    chat_history = []
    # Limit to the last 2 messages (1 QA pair) to prevent CPU overload
    for msg in history[-2:]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["text"]))
        else:
            chat_history.append(AIMessage(content=msg["text"]))

    try:
        result  = rag_chain.invoke({"input": message, "chat_history": chat_history})
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
