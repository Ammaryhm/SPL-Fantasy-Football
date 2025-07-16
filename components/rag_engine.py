import json
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data/")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# -----------------------------
# Load all .json and .csv files
# -----------------------------
def load_documents():
    docs = []

    for file in DATA_DIR.glob("*.json"):
        loader = JSONLoader(
            file_path=str(file),
            jq_schema=".",  # Optional: refine to pull specific fields
            text_content=False
        )
        docs.extend(loader.load())

    return docs

# -----------------------------
# Build VectorStore from docs
# -----------------------------
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

# -----------------------------
# Exported RAG retriever
# -----------------------------
docs = load_documents()
vectorstore = build_vectorstore(docs)
retriever = vectorstore.as_retriever()
