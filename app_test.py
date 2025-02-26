
from graphs.rag2.workflow import get_compiled_graph
from database.initialize_db import get_db
from langchain_chroma import Chroma
import chromadb

import os
from dotenv import load_dotenv
import pprint

load_dotenv()

LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

workflow = None

# Here goes flask app
#
#
#
#################

def initialize_components():

    global workflow

    workflow = get_compiled_graph()

    # return workflow

if __name__ == "__main__":
    try:
        initialize_components()
    except:
        print("fail first")
    print("Se ejecuto")