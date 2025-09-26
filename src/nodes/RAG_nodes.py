from src.states.RAGState import GraphState , RAG
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage 
from src.retrievers.retriever import retriever 
from src.chains.rag_chain import rag_chain
from src.chains.retrieval_grader import retrieval_grader
from src.chains.question_rewriter import question_rewriter
from src.web_search.web_search_tool import web_search_tool
from src.chains.question_router import question_router
from src.chains.answer_grader import answer_grader
from src.chains.hallucination_grader import GradeHallucinations

class RAG_nodes: 
    """
    a class to represent RAG nodes
    """

    def __init__(self ):
        self.retriever = retriever().get_retriever()
        self.rag_chain = rag_chain().get_rag_chain()
        self.retrieval_grader = retrieval_grader().get_retrieval_grader()
        self.question_rewriter = question_rewriter().question_rewriter()
        self.web_search_tool = web_search_tool().get_web_search_tool()
        self.question_router = question_router().get_question_router()
        self.hallucination_grader = GradeHallucinations().get_hallucination_grader()
        self.answer_grader = answer_grader().get_answer_grader()
        

    def retrieve(self , state: GraphState):

        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    

    def generate(self , state: GraphState): 
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    

    def grade_documents(self, state: GraphState):
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
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            # Handle both dict and Pydantic model responses
            grade = score.binary_score if hasattr(score, 'binary_score') else score.get('binary_score', 'no')
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}
    


    def transform_query(self , state: GraphState):
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
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}
    


    def web_search(self , state: GraphState):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}
    


    ### edges 

    def route_question(self , state: GraphState):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        # Handle both dict and Pydantic model responses
        datasource = source.datasource if hasattr(source, 'datasource') else source.get('datasource', 'vectorstore')
        
        if datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        


    def decide_to_generate(self, state: GraphState):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
        


        
    def grade_generation_v_documents_and_question(self , state:GraphState ):
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

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        # Handle both dict and Pydantic model responses
        grade = score.binary_score if hasattr(score, 'binary_score') else score.get('binary_score', 'no')

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            # Handle both dict and Pydantic model responses
            grade = score.binary_score if hasattr(score, 'binary_score') else score.get('binary_score', 'no')
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        





        


            








    


        


    


    
