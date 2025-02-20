from langchain_chroma import Chroma
import chromadb
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()


COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_DB_OLLAMA_PATH = os.getenv("VECTOR_DB_OLLAMA_PATH")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")


def get_db():

    chroma_client = chromadb.PersistentClient(VECTOR_DB_OLLAMA_PATH)

    vectorstore = Chroma(
    client = chroma_client,
    collection_name = COLLECTION_NAME,
    embedding_function = OllamaEmbeddings(model = LLM_MODEL_NAME)
    )

    return vectorstore