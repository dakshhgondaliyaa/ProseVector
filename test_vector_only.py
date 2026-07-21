"""Test ONLY the FAISS vector retrieval accuracy — no LLM involved."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

TEST_QUERIES = [
    {
        "category": "Economics",
        "query": "An economics text that discusses the division of labor, productivity, and free markets, arguing that an 'invisible hand' guides self-interest to benefit society.",
        "expected_title": "The Wealth of Nations"
    },
    {
        "category": "Political Philosophy",
        "query": "A political text arguing that society requires a strong, absolute sovereign to avoid the 'war of all against all' and ensure order.",
        "expected_title": "Leviathan"
    },
    {
        "category": "Political Science",
        "query": "A French sociologist's observations on the American political system, equality, and civil society after visiting the United States in the 1830s.",
        "expected_title": "Democracy in America"
    },
    {
        "category": "Anthropology / History",
        "query": "An anthropological history book arguing that environmental and geographic factors, rather than intellectual or genetic superiority, allowed Eurasian societies to conquer others.",
        "expected_title": "Guns, Germs, and Steel"
    },
    {
        "category": "Philosophy of Science",
        "query": "A philosophy of science book that introduces the concept of a 'paradigm shift', arguing that science progresses through sudden revolutions rather than linear accumulation of facts.",
        "expected_title": "The Structure of Scientific Revolutions"
    },
    {
        "category": "Ethics",
        "query": "An ethical theory book arguing that the best action is the one that maximizes overall happiness or pleasure for the greatest number of people.",
        "expected_title": "Utilitarianism"
    },
]

print("=" * 60)
print("  FAISS Vector Retrieval Accuracy Test (No LLM)")
print("=" * 60)

passed = 0
total = len(TEST_QUERIES)

for item in TEST_QUERIES:
    docs = vectorstore.similarity_search(item["query"], k=TOP_K)
    
    # Check if expected title appears in top-k results
    retrieved_titles = [doc.metadata.get("title", "") for doc in docs]
    unique_titles = list(dict.fromkeys(retrieved_titles))  # preserve order, remove dupes
    
    found = item["expected_title"] in retrieved_titles
    rank = retrieved_titles.index(item["expected_title"]) + 1 if found else -1
    
    status = "PASS" if found else "FAIL"
    if found:
        passed += 1
    
    print(f"\n[{status}] {item['category']}")
    print(f"  Expected: {item['expected_title']}")
    print(f"  Rank: {'#' + str(rank) if found else 'NOT FOUND in top ' + str(TOP_K)}")
    print(f"  Top 5 retrieved: {unique_titles[:5]}")

print(f"\n{'=' * 60}")
print(f"  FAISS Retrieval Accuracy: {passed}/{total} ({100*passed/total:.1f}%)")
print(f"{'=' * 60}")
