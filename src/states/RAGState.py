from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field

class RAG(BaseModel) : 

    answer :str = Field(description="The answer to the question")
    source_documents : List[str] = Field(description="The source documents used to generate the answer")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    number_of_document_tries: int
    documents: List[str]
    upload_status: str
    source_type: str  # "vectorstore" or "websearch"



    


