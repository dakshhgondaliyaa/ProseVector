import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datasets import load_dataset
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATASET_NAME       = "textminr/cmu-book-summaries"
DATASET_SPLIT      = "train"
TEXT_COLUMN_TITLE   = "title"
TEXT_COLUMN_SUMMARY = "summary"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE         = 2000
CHUNK_OVERLAP      = 200
FAISS_INDEX_PATH   = "faiss_index"
MAX_BOOKS          = 20000

def load_books():
    print("Loading CMU Book Summaries dataset from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"Total books available: {len(dataset)}")
    return dataset

def preprocess_and_chunk(dataset):
    print("Preprocessing and chunking book entries...")
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    docs = []
    for i, item in enumerate(dataset):
        if i >= MAX_BOOKS:
            break
        title   = item.get(TEXT_COLUMN_TITLE, "Unknown Title")
        author  = item.get("author", "Unknown Author")
        summary = item.get(TEXT_COLUMN_SUMMARY, "")
        if summary and len(summary) > 50:
            summary_chunks = splitter.split_text(summary)
            for chunk in summary_chunks:
                combined = f"Book Title: {title}\nAuthor: {author}\nPlot Summary: {chunk}"
                docs.append(Document(page_content=combined, metadata={"title": title, "author": author}))
    print(f"Total chunks created: {len(docs)}")
    return docs

def build_and_save_index(chunks):
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Creating FAISS vector index (takes 2-5 mins)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to ./{FAISS_INDEX_PATH}/")
    return vectorstore

if __name__ == "__main__":
    if os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index already exists. Delete faiss_index/ to rebuild.")
    else:
        dataset    = load_books()
        chunks     = preprocess_and_chunk(dataset)
        build_and_save_index(chunks)
        print("\nSetup complete! Now run: python app.py")
