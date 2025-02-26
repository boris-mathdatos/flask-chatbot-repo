from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")