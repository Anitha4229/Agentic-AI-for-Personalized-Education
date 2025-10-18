# ingest.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings      # ✅ new import
from langchain_community.vectorstores import Chroma

load_dotenv()  # Load environment variables from .env file

# 1. Define the folder for persistent storage
PERSIST_DIRECTORY = 'db'

# 2. Define the data source folder
SOURCE_DIRECTORY = 'data'

def ingest_data():
    print("Starting data ingestion...")

    # Load all documents from the data directory
    documents = []
    for file in os.listdir(SOURCE_DIRECTORY):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(SOURCE_DIRECTORY, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages from PDF files.")

    # 3. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text.")

    # 4. Create embeddings (convert text chunks into vectors)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. Store embeddings in ChromaDB
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    # ✅ No need for vectordb.persist() — Chroma ≥0.4 does it automatically
    vectordb = None  # Clear from memory

    print("Ingestion complete! Data has been stored in the vector database.")

if __name__ == "__main__":
    ingest_data()
