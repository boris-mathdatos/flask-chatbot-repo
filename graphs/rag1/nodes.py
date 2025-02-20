from .chains import rag_chain, retrieval_grader_chain, question_rewriter_chain
from database.initialize_db import get_db

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(model = os.getenv("LLM_MODEL_NAME"))


def retrieve_documents(state):

    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    doc_count = state["contador_relevant_docs"]

    if doc_count == None:
        doc_count = 1
    else:
        doc_count += 1

    # Retrieval
    retriever = get_db().as_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "contador_relevant_docs": doc_count}


def generate(state):

    global llm

    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]
    contador = state["contador"]

    if contador == None:
        contador = 1
    else:
        contador += 1


    # RAG generation
    generation = rag_chain(llm).invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "contador": contador}


def grade_documents(state):

    global llm

    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader_chain(llm).invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT IS RELEVANT---")
            print(f"------------- Relevant Document --------")
            print(d.page_content)
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT IS NOT RELEVANT---")
            print(f"------------- Not Relevant Document --------")
            print(d.page_content)
            continue
    doc_count = state["contador_relevant_docs"]
    print(f"************* doc count en el grader ***** : {doc_count}")
    
    return {"documents": filtered_docs, "question": question, "contador_relevant_docs": doc_count + 1}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter_chain(llm).invoke({"question": question})
    return {"documents": documents, "question": better_question}