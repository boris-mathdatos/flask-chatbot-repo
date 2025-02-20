from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field, BaseModel
from typing import Literal

import os
from dotenv import load_dotenv

# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[2]))

load_dotenv()

LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")


# Chain

def rag_chain(llm):
    system = """You are an specialist in companies finantial health, mathematics and also provide finantial assistance. / Supported by a set of retrieved facts. \n 
        Respond the question asked the best way you can."""
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of retrieved facts: \n\n {context} \n\n Question to respond: {question}"),
        ]
    )

    return rag_prompt | llm | StrOutputParser()

def retrieval_grader_chain(llm):
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description = "Documents are relevant to the question, 'yes' or 'no'"
        )


    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    return grade_prompt | structured_llm_grader

def question_rewriter_chain(llm):

    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    return re_write_prompt | llm | StrOutputParser()


def question_router_chain(llm):

    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["vectorstore", "web_search"] = Field(
            ...,
            description = "Given a user question choose to route it to web search or a vectorstore.",
        )


    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to mathematics and technical question.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    return route_prompt | structured_llm_router

def hallucination_grader_chain(llm):
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description = "Answer is grounded in the facts, 'yes' or 'no'"
        )

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    return hallucination_prompt | structured_llm_grader



def answer_grader_chain(llm):
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description = "Answer addresses the question, 'yes' or 'no'"
    )


    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    return answer_prompt | structured_llm_grader