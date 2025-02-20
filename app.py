from flask import (Flask,
                   request,
                   render_template,
                   jsonify,
                   redirect,
                   url_for)

from forms import ChatForm
from config import Config
from graphs.rag1.workflow import get_compiled_graph
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


app = Flask(__name__)
app.config.from_object(Config)

@app.route("/", methods = ["GET", "POST"])
def index():
    
    form = ChatForm()
    if form.validate_on_submit():
        user_message = form.message.data
        # bot_response = get_bot_response(user_message)
        response = workflow.invoke({"question": user_message})
        return render_template('index.html', form = form, user_message = user_message, bot_response = response.get("generation", "NO ESTA EN RESP DICT"))
    return render_template('index.html', form = form)

    

def initialize_components():

    global workflow
    "Docstring"
    
    workflow = get_compiled_graph()

    
initialize_components()
# result = workflow.invoke({"question": "what is an agent?"})
# print(result.get("generation", "NDA tie"))


if __name__ == "__main__":
    app.run(debug = True)

    # from graphs.rag1.chains import hallucination_grader_chain
    # from langchain_ollama import ChatOllama

    # llm = ChatOllama(model = "llama3.1")
    # chain = hallucination_grader_chain(llm)
    # print(f"{chain =}")
    # print(chain.invoke({"documents": "error", "generation": "aaaa"}))