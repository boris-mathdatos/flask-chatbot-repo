from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict

from graphs.rag2.nodes import *
from .edges import *

def get_compiled_graph():

    # States
    class GraphState(TypedDict):
        a: str
        b: str

    workflow = StateGraph(GraphState)

    # Nodes


    # Building Graph

    return workflow.compile()