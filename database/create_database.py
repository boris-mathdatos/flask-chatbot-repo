import argparse
import sys
import json
import shutil
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from embeddings.embeddings import Embeddings
from langchain.schema import Document
from langchain_chroma import Chroma
import chromadb 

from data.urls import urls_list

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

VECTOR_DB_OLLAMA_PATH = os.getenv("VECTOR_DB_OLLAMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

def main():
    # Checking whether the database should be cleared (--clear)
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", nargs = "?", const = "both", choices = ["ollama", "both"], help = "Reset the database")
    parser.add_argument("--embedding-model", type = str, default = "ollama", help = "The embedding model to use (ollama embbeddings for now)")
    args = parser.parse_args()

    if args.delete:
        delete_database(args.delete)
        return

    embeddings = Embeddings(model_name = args.embedding_model, api_key = None)
    emb = embeddings.get_embedding_function()

    if args.embedding_model == "ollama":
        db_path = VECTOR_DB_OLLAMA_PATH
    else:
        raise ValueError("Currently supports Ollama embeddings only")

    chroma_client = chromadb.PersistentClient(db_path)
        
    urls = urls_list
    documents = load_web_documents(urls = urls)
    doc_splits = split_documents(documents = documents)

    print("Creating DB...")
    vectorstore = Chroma.from_documents(
        client = chroma_client,
        documents = doc_splits,
        collection_name = COLLECTION_NAME,
        embedding = emb
    )
    print("Done...")

def load_web_documents(urls: list[str]):
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]
    
def split_documents(documents : list[Document]):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)

def delete_database(embedding_model):
    if embedding_model == "ollama":
        db_path = VECTOR_DB_OLLAMA_PATH
    else:
        raise ValueError("Unsupported embedding model specified.")
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    print("DB deleted...")

if __name__ == "__main__":
    main()