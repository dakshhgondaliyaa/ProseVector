"""
Step 1: TOP_K comparison test (3 vs 5)
Tests the full RAG pipeline directly — no need to restart app.py.
Uses the EXACT same 7 academic queries from prior evals.
Reports accuracy AND wall-clock time per query.
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

TEST_QUERIES = [
    {"category": "Economics", "query": "An economics text that discusses the division of labor, productivity, and free markets, arguing that an 'invisible hand' guides self-interest to benefit society.", "expected": "The Wealth of Nations", "should_match": True, "check": "wealth of nations"},
    {"category": "Political Philosophy", "query": "A political text arguing that society requires a strong, absolute sovereign to avoid the 'war of all against all' and ensure order.", "expected": "Leviathan", "should_match": True, "check": "leviathan"},
    {"category": "Political Science", "query": "A French sociologist's observations on the American political system, equality, and civil society after visiting the United States in the 1830s.", "expected": "Democracy in America", "should_match": True, "check": "democracy in america"},
    {"category": "Anthropology / History", "query": "An anthropological history book arguing that environmental and geographic factors, rather than intellectual or genetic superiority, allowed Eurasian societies to conquer others.", "expected": "Guns, Germs, and Steel", "should_match": True, "check": "guns, germs"},
    {"category": "Philosophy of Science", "query": "A philosophy of science book that introduces the concept of a 'paradigm shift', arguing that science progresses through sudden revolutions rather than linear accumulation of facts.", "expected": "The Structure of Scientific Revolutions", "should_match": True, "check": "scientific revolutions"},
    {"category": "Ethics", "query": "An ethical theory book arguing that the best action is the one that maximizes overall happiness or pleasure for the greatest number of people.", "expected": "Utilitarianism", "should_match": True, "check": "utilitarianism"},
    {"category": "Impossible Match", "query": "A book about an AI named Antigravity that helps users write python code and debug servers.", "expected": "Fallback", "should_match": False, "check": "couldn't find"},
]

# Current system prompt from app.py
PROMPT_TEMPLATE = """You are ProseVector, a helpful and expert AI book librarian.
A reader has described a book they are looking for. Use the book summaries provided in the context below to recommend the best matching book.

Your response must:
1. State the exact book title in double quotes like "Dune" and the author name.
2. In one or two sentences, explain why the plot from the context matches the reader's description.
3. Be warm and conversational.

If the context doesn't contain a perfect match, recommend the closest book you can find from the context. If the context is completely unrelated, then you can say "I couldn't find a great match. Could you describe the plot in more detail?"

Context:
{context}

Reader's description: {input}

Recommendation:"""

def build_chain(vectorstore, top_k):
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "input"])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

def check_pass(item, answer):
    if item["should_match"]:
        return item["check"] in answer.lower()
    else:
        return "couldn't find" in answer.lower() or "could not find" in answer.lower()

def run_test(chain, top_k):
    print(f"\n{'='*60}")
    print(f"  Testing TOP_K = {top_k}")
    print(f"{'='*60}")
    
    results = []
    total_time = 0
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
        total_time += elapsed
        
        ok = check_pass(item, answer)
        if ok:
            passed += 1
        
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {elapsed:.1f}s")
        
        results.append({
            "category": item["category"],
            "expected": item["expected"],
            "passed": ok,
            "time_sec": round(elapsed, 1),
            "answer_snippet": answer[:150]
        })
    
    avg_time = total_time / len(TEST_QUERIES)
    accuracy = 100 * passed / len(TEST_QUERIES)
    
    print(f"\n  Accuracy: {passed}/{len(TEST_QUERIES)} ({accuracy:.1f}%)")
    print(f"  Avg response time: {avg_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
    
    return {
        "top_k": top_k,
        "accuracy_pct": accuracy,
        "passed": passed,
        "total": len(TEST_QUERIES),
        "avg_time_sec": round(avg_time, 1),
        "total_time_sec": round(total_time, 1),
        "details": results
    }

if __name__ == "__main__":
    print("Loading FAISS index and embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Test TOP_K=3 first (our current baseline)
    chain3 = build_chain(vectorstore, top_k=3)
    results_3 = run_test(chain3, top_k=3)
    
    # Test TOP_K=5
    chain5 = build_chain(vectorstore, top_k=5)
    results_5 = run_test(chain5, top_k=5)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: TOP_K=3 vs TOP_K=5")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'TOP_K=3':>10} {'TOP_K=5':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Accuracy':<25} {results_3['accuracy_pct']:>9.1f}% {results_5['accuracy_pct']:>9.1f}%")
    print(f"  {'Avg response time':<25} {results_3['avg_time_sec']:>9.1f}s {results_5['avg_time_sec']:>9.1f}s")
    print(f"  {'Total time':<25} {results_3['total_time_sec']:>9.1f}s {results_5['total_time_sec']:>9.1f}s")
    
    winner = "TOP_K=3" if results_3['accuracy_pct'] >= results_5['accuracy_pct'] else "TOP_K=5"
    if results_3['accuracy_pct'] == results_5['accuracy_pct']:
        winner = "TOP_K=3 (same accuracy, faster)"
    print(f"\n  Winner: {winner}")
    
    # Save full results
    with open("step1_topk_comparison.json", "w") as f:
        json.dump({"top_k_3": results_3, "top_k_5": results_5}, f, indent=2)
    print(f"\n  Full results saved to step1_topk_comparison.json")
