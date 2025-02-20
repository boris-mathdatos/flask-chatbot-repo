import pprint
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(model = os.getenv("LLM_MODEL_NAME"))

from graphs.rag1.chains import (question_router_chain,
                                hallucination_grader_chain,
                                answer_grader_chain)



def route_question(state):

    global llm

    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router_chain(llm).invoke({"question": question})


    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        # return "web_search"
        return "vectorstore"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    # state["question"]
    doc_count = state["contador_relevant_docs"]
    filtered_documents = state["documents"]

    print(f"********  doc count **** : {doc_count}")


    if doc_count >= 2:
        return "generate"

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATING WITH RELEVANT DOCUMENTS---")
        return "generate"


def grade_generation_v_documents_and_question(state):

    global llm

    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]



    score = hallucination_grader_chain(llm).invoke(
        {"documents": documents, "generation": generation}
    )

    try:
        grade = score.binary_score
    except:
        grade = "no"


    ######

    contador = state["contador"]
    doc_count = state["contador_relevant_docs"]

    print(f"DESDE EL CHECK HALLUCINATION: contador = {contador}")
    if contador >= 2 or doc_count >= 2:
        return "useful"
    ######

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader_chain(llm).invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        state["contador"] = state["contador"] + 1
        return "not supported"