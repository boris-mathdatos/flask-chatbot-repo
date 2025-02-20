from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from typing import List
from graphs.rag1.nodes import (retrieve_documents,
                   generate,
                   grade_documents,
                   transform_query)
from .edges import (route_question,
                   decide_to_generate,
                   grade_generation_v_documents_and_question)


def get_compiled_graph():
    class GraphState(TypedDict):

        question: str
        generation: str
        documents: List[str]
        contador: int
        contador_relevant_docs : int

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve_documents)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore": "retrieve",
        },
    )
    # workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()