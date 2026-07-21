"""
Step 2: Fallback prompt tightening test.
Uses the NEW tightened prompt. Tests all 7 queries to verify:
  - The 5 previously-passing queries still pass (no over-correction)
  - The "Impossible Match" now correctly returns fallback
"""
import time
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3"
TOP_K = 3  # Winner from Step 1

TEST_QUERIES = [
    {"category": "Economics", "query": "An economics text that discusses the division of labor, productivity, and free markets, arguing that an 'invisible hand' guides self-interest to benefit society.", "expected": "The Wealth of Nations", "should_match": True, "check": "wealth of nations"},
    {"category": "Political Philosophy", "query": "A political text arguing that society requires a strong, absolute sovereign to avoid the 'war of all against all' and ensure order.", "expected": "Leviathan", "should_match": True, "check": "leviathan"},
    {"category": "Political Science", "query": "A French sociologist's observations on the American political system, equality, and civil society after visiting the United States in the 1830s.", "expected": "Democracy in America", "should_match": True, "check": "democracy in america"},
    {"category": "Anthropology / History", "query": "An anthropological history book arguing that environmental and geographic factors, rather than intellectual or genetic superiority, allowed Eurasian societies to conquer others.", "expected": "Guns, Germs, and Steel", "should_match": True, "check": "guns, germs"},
    {"category": "Philosophy of Science", "query": "A philosophy of science book that introduces the concept of a 'paradigm shift', arguing that science progresses through sudden revolutions rather than linear accumulation of facts.", "expected": "The Structure of Scientific Revolutions", "should_match": True, "check": "scientific revolutions"},
    {"category": "Ethics", "query": "An ethical theory book arguing that the best action is the one that maximizes overall happiness or pleasure for the greatest number of people.", "expected": "Utilitarianism", "should_match": True, "check": "utilitarianism"},
    {"category": "Impossible Match", "query": "A book about an AI named Antigravity that helps users write python code and debug servers.", "expected": "Fallback", "should_match": False, "check": "couldn't find"},
]

# NEW tightened prompt from app.py
PROMPT_TEMPLATE = """You are ProseVector, a helpful and expert AI book librarian.
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

def check_pass(item, answer):
    if item["should_match"]:
        return item["check"] in answer.lower()
    else:
        return "couldn't find" in answer.lower() or "could not find" in answer.lower()

if __name__ == "__main__":
    print("Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "input"])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print(f"\n{'='*60}")
    print(f"  Step 2: Tightened Fallback Prompt Test (TOP_K={TOP_K})")
    print(f"{'='*60}")
    
    results = []
    passed = 0
    
    for item in TEST_QUERIES:
        print(f"  [{item['category']}] ...", end=" ", flush=True)
        start = time.time()
        try:
            result = chain.invoke({"input": item["query"]})
            answer = result["answer"]
        except Exception as e:
            answer = f"ERROR: {e}"
        elapsed = time.time() - start
        
        ok = check_pass(item, answer)
        if ok:
            passed += 1
        
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {elapsed:.1f}s")
        if not ok:
            print(f"    Answer: {answer[:200]}")
        
        results.append({
            "category": item["category"],
            "expected": item["expected"],
            "passed": ok,
            "time_sec": round(elapsed, 1),
            "answer": answer
        })
    
    accuracy = 100 * passed / len(TEST_QUERIES)
    print(f"\n  Accuracy: {passed}/{len(TEST_QUERIES)} ({accuracy:.1f}%)")
    
    # Check for regressions
    previously_passing = ["Political Science", "Anthropology / History", "Philosophy of Science", "Ethics"]
    regressions = [r for r in results if r["category"] in previously_passing and not r["passed"]]
    if regressions:
        print(f"\n  ⚠️  REGRESSIONS DETECTED in: {[r['category'] for r in regressions]}")
    else:
        print(f"\n  ✅ No regressions — all previously-passing queries still pass")
    
    # Check impossible match specifically
    impossible = [r for r in results if r["category"] == "Impossible Match"][0]
    if impossible["passed"]:
        print(f"  ✅ Impossible Match now correctly returns fallback!")
    else:
        print(f"  ❌ Impossible Match still not returning fallback")
        print(f"     Got: {impossible['answer'][:200]}")
    
    with open("step2_fallback_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to step2_fallback_results.json")
