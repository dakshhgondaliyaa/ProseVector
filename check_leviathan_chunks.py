import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

docs = vectorstore.similarity_search("Leviathan", k=10)
print("Looking for Leviathan chunks...")
found = 0
for d in docs:
    if d.metadata.get("title") == "Leviathan":
        print("--- CHUNK ---")
        print(d.page_content[:200])
        found += 1
if found == 0:
    print("Could not find Leviathan chunks!")
else:
    print(f"Found {found} chunks.")
