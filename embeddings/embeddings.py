from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class Embeddings:
    def __init__(self, model_name, api_key = None):
        self.model_name = model_name
        self.api_key = api_key

    def get_embedding_function(self):
        if self.model_name == "ollama":
            return OllamaEmbeddings(model = os.getenv("LLAMA_MODEL"))