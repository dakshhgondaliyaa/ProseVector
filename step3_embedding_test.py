"""
Step 3: Embedding model upgrade test.
Swaps all-MiniLM-L6-v2 (22M) for all-mpnet-base-v2 (110M).
Tests ONLY FAISS retrieval (no LLM) to isolate the embedding variable.
Reports:
  - Retrieval accuracy for all 6 academic books
  - Whether Leviathan now ranks #1
  - Wall-clock latency per query (embedding + search time)
  - Comparison with MiniLM latency
"""
import time
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from datasets import load_dataset

FAISS_INDEX_PATH = "faiss_index"
OLD_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEW_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 3

TEST_QUERIES = [
    {"category": "Economics", "query": "An economics text that discusses the division of labor, productivity, and free markets, arguing that an 'invisible hand' guides self-interest to benefit society.", "expected_title": "The Wealth of Nations"},
    {"category": "Political Philosophy", "query": "A political text arguing that society requires a strong, absolute sovereign to avoid the 'war of all against all' and ensure order.", "expected_title": "Leviathan"},
    {"category": "Political Science", "query": "A French sociologist's observations on the American political system, equality, and civil society after visiting the United States in the 1830s.", "expected_title": "Democracy in America"},
    {"category": "Anthropology / History", "query": "An anthropological history book arguing that environmental and geographic factors, rather than intellectual or genetic superiority, allowed Eurasian societies to conquer others.", "expected_title": "Guns, Germs, and Steel"},
    {"category": "Philosophy of Science", "query": "A philosophy of science book that introduces the concept of a 'paradigm shift', arguing that science progresses through sudden revolutions rather than linear accumulation of facts.", "expected_title": "The Structure of Scientific Revolutions"},
    {"category": "Ethics", "query": "An ethical theory book arguing that the best action is the one that maximizes overall happiness or pleasure for the greatest number of people.", "expected_title": "Utilitarianism"},
]

TARGET_BOOKS = [
    "The Wealth of Nations",
    "Leviathan",
    "Democracy in America",
    "Guns, Germs, and Steel",
    "The Structure of Scientific Revolutions",
    "Utilitarianism"
]

def test_retrieval(vectorstore, model_name, embeddings_obj):
    """Run retrieval test and return results with timing."""
    print(f"\n{'='*60}")
    print(f"  Retrieval Test: {model_name.split('/')[-1]}")
    print(f"{'='*60}")
    
    results = []
    passed = 0
    total_query_time = 0
    
    for item in TEST_QUERIES:
        start = time.time()
        docs = vectorstore.similarity_search(item["query"], k=TOP_K)
        elapsed = time.time() - start
        total_query_time += elapsed
        
        retrieved_titles = [doc.metadata.get("title", "") for doc in docs]
        unique_titles = list(dict.fromkeys(retrieved_titles))
        
        found = item["expected_title"] in retrieved_titles
        rank = retrieved_titles.index(item["expected_title"]) + 1 if found else -1
        
        if found:
            passed += 1
        
        status = "PASS" if found else "FAIL"
        print(f"  [{status}] {item['category']} ({elapsed*1000:.0f}ms)")
        print(f"    Expected: {item['expected_title']}")
        print(f"    Rank: {'#' + str(rank) if found else 'NOT FOUND in top ' + str(TOP_K)}")
        print(f"    Top results: {unique_titles}")
        
        results.append({
            "category": item["category"],
            "expected": item["expected_title"],
            "found": found,
            "rank": rank,
            "latency_ms": round(elapsed * 1000),
            "top_results": unique_titles
        })
    
    avg_latency = (total_query_time / len(TEST_QUERIES)) * 1000
    accuracy = 100 * passed / len(TEST_QUERIES)
    
    print(f"\n  Accuracy: {passed}/{len(TEST_QUERIES)} ({accuracy:.1f}%)")
    print(f"  Avg query latency: {avg_latency:.0f}ms")
    
    return {
        "model": model_name,
        "accuracy_pct": accuracy,
        "passed": passed,
        "total": len(TEST_QUERIES),
        "avg_latency_ms": round(avg_latency),
        "details": results
    }

if __name__ == "__main__":
    # --- Test 1: Old model (MiniLM) with existing index ---
    print("Loading OLD embedding model (MiniLM)...")
    old_start = time.time()
    old_embeddings = HuggingFaceEmbeddings(
        model_name=OLD_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    old_load_time = time.time() - old_start
    print(f"  Model load time: {old_load_time:.1f}s")
    
    old_vectorstore = FAISS.load_local(FAISS_INDEX_PATH, old_embeddings, allow_dangerous_deserialization=True)
    old_results = test_retrieval(old_vectorstore, OLD_MODEL, old_embeddings)
    
    # --- Rebuild index with new model (mpnet) ---
    print(f"\n{'='*60}")
    print(f"  Rebuilding index with NEW model (mpnet)...")
    print(f"{'='*60}")
    
    print("Loading NEW embedding model (mpnet)...")
    new_start = time.time()
    new_embeddings = HuggingFaceEmbeddings(
        model_name=NEW_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    new_load_time = time.time() - new_start
    print(f"  Model load time: {new_load_time:.1f}s")
    
    # Rebuild the index with the same data but new embeddings
    print("Loading dataset...")
    dataset = load_dataset("textminr/cmu-book-summaries", split="train")
    
    selected_items = []
    target_found = set()
    for item in dataset:
        title = item.get("title", "")
        if title in TARGET_BOOKS and title not in target_found:
            selected_items.append(item)
            target_found.add(title)
        if len(selected_items) < 100 and title not in TARGET_BOOKS:
            selected_items.append(item)
    
    print(f"  Found {len(target_found)}/{len(TARGET_BOOKS)} target books")
    print(f"  Total books for index: {len(selected_items)}")
    
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2000, chunk_overlap=200)
    docs = []
    for item in selected_items:
        title = item.get("title", "Unknown")
        author = item.get("author", "Unknown")
        summary = item.get("summary", "")
        if summary and len(summary) > 50:
            for chunk in splitter.split_text(summary):
                combined = f"Book Title: {title}\nAuthor: {author}\nPlot Summary: {chunk}"
                docs.append(Document(page_content=combined, metadata={"title": title, "author": author}))
    
    print(f"  Total chunks: {len(docs)}")
    
    index_start = time.time()
    new_vectorstore = FAISS.from_documents(docs, new_embeddings)
    index_build_time = time.time() - index_start
    print(f"  Index build time: {index_build_time:.1f}s")
    
    # Save as a separate index so we don't overwrite the original
    new_vectorstore.save_local("faiss_index_mpnet")
    
    # --- Test 2: New model (mpnet) ---
    new_results = test_retrieval(new_vectorstore, NEW_MODEL, new_embeddings)
    
    # --- Comparison ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON: MiniLM vs mpnet")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'MiniLM':>10} {'mpnet':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Retrieval Accuracy':<30} {old_results['accuracy_pct']:>9.1f}% {new_results['accuracy_pct']:>9.1f}%")
    print(f"  {'Avg query latency':<30} {old_results['avg_latency_ms']:>8}ms {new_results['avg_latency_ms']:>8}ms")
    print(f"  {'Model load time':<30} {old_load_time:>9.1f}s {new_load_time:>9.1f}s")
    print(f"  {'Index build time':<30} {'(existing)':>10} {index_build_time:>9.1f}s")
    
    # Leviathan check
    old_lev = [r for r in old_results["details"] if r["category"] == "Political Philosophy"][0]
    new_lev = [r for r in new_results["details"] if r["category"] == "Political Philosophy"][0]
    print(f"\n  Leviathan fix check:")
    print(f"    MiniLM: {'Found at #' + str(old_lev['rank']) if old_lev['found'] else 'NOT FOUND -> ' + str(old_lev['top_results'])}")
    print(f"    mpnet:  {'Found at #' + str(new_lev['rank']) if new_lev['found'] else 'NOT FOUND -> ' + str(new_lev['top_results'])}")
    
    # Check for regressions
    regressions = []
    for old_r, new_r in zip(old_results["details"], new_results["details"]):
        if old_r["found"] and not new_r["found"]:
            regressions.append(new_r["category"])
    
    if regressions:
        print(f"\n  WARNING: Regressions in: {regressions}")
    else:
        print(f"\n  No regressions detected")
    
    with open("step3_embedding_results.json", "w") as f:
        json.dump({"miniLM": old_results, "mpnet": new_results}, f, indent=2)
    print(f"\n  Full results saved to step3_embedding_results.json")
