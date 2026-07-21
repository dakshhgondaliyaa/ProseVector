import os
from datasets import load_dataset
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATASET_NAME = "textminr/cmu-book-summaries"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

TARGET_BOOKS = [
    "The Wealth of Nations",
    "Leviathan",
    "Democracy in America",
    "Guns, Germs, and Steel",
    "The Structure of Scientific Revolutions",
    "Utilitarianism"
]

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

print("Finding target books...")
selected_items = []
target_found = set()
for item in dataset:
    title = item.get("title", "")
    if title in TARGET_BOOKS and title not in target_found:
        selected_items.append(item)
        target_found.add(title)
        
    if len(selected_items) < 100 and title not in TARGET_BOOKS:
        selected_items.append(item)

print(f"Found {len(target_found)} out of {len(TARGET_BOOKS)} target books.")
print(f"Total books for test index: {len(selected_items)}")

print("Chunking...")
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

print("Embedding and building index...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print("Index built successfully!")
