"""Rebuild index with tagged Leviathan, then test ONLY the Leviathan query."""
import time
import subprocess
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Rebuild index
print("Rebuilding FAISS index with tagged Leviathan...")
subprocess.run(["python", "fast_test_accuracy.py"], check=True)

# Step 2: Load and test
print("\nLoading rebuilt index...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

query = "A political text arguing that society requires a strong, absolute sovereign to avoid the 'war of all against all' and ensure order."

print(f"\nQuery: {query}")
print(f"\nBefore fix: Leviathan was NOT FOUND in top 3 (returned On War)")

start = time.time()
docs = vectorstore.similarity_search(query, k=3)
elapsed = time.time() - start

retrieved_titles = [doc.metadata.get("title", "") for doc in docs]
unique_titles = list(dict.fromkeys(retrieved_titles))

found = "Leviathan" in retrieved_titles
rank = retrieved_titles.index("Leviathan") + 1 if found else -1

print(f"After fix:  {'Leviathan found at #' + str(rank) if found else 'Leviathan still NOT FOUND'}")
print(f"Top results: {unique_titles}")
print(f"Query time: {elapsed*1000:.0f}ms")

if found:
    print(f"\nResult: FIXED - Leviathan went from NOT FOUND to #{rank}")
else:
    print(f"\nResult: NOT FIXED - still returning {unique_titles}")
