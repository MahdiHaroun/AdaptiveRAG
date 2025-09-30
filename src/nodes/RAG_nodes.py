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
from utils.generated_document_uploader import upload_generated_answers

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
        attempts = state.get("number_of_document_tries", 0)
        upload_status = state.get("upload_status", "False")
        
        # Retrieval
        documents = self.retriever.invoke(question)
        return {
            "documents": documents, 
            "question": question, 
            "number_of_document_tries": attempts, 
            "upload_status": upload_status,
            "source_type": "vectorstore"
        }
    

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
        attempts = state.get("number_of_document_tries", 0)
        upload_status = state.get("upload_status", "False")
        source_type = state.get("source_type", "unknown")

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation, 
            "number_of_document_tries": attempts, 
            "upload_status": upload_status,
            "source_type": source_type
        }
    

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
        attempts = state.get("number_of_document_tries", 0)
        upload_status = state.get("upload_status", "False")
        source_type = state.get("source_type", "unknown")

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
        return {
            "documents": filtered_docs, 
            "question": question, 
            "number_of_document_tries": attempts, 
            "upload_status": upload_status,
            "source_type": source_type
        }
    


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
        current_attempts = state.get("number_of_document_tries", 0)
        upload_status = state.get("upload_status", "False")
        source_type = state.get("source_type", "unknown")

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        
        # Increment the number of attempts
        new_attempts = current_attempts + 1
        print(f"---INCREMENTING ATTEMPTS: {new_attempts}---")
        
        return {
            "documents": documents, 
            "question": better_question, 
            "number_of_document_tries": new_attempts, 
            "upload_status": upload_status,
            "source_type": source_type
        }
    


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
        attempts = state.get("number_of_document_tries", 0)
        upload_status = state.get("upload_status", "False")

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {
            "documents": web_results, 
            "question": question, 
            "number_of_document_tries": attempts, 
            "upload_status": upload_status,
            "source_type": "websearch"
        }
    


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
        
        
    def grade_generation_v_documents_and_question(self , state:GraphState ):
        """
        Determines whether the generation is grounded in the document and answers question.
        Only routes to human-in-the-loop for web search results.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]  
        generation = state["generation"]
        source_type = state.get("source_type", "unknown")

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
                # Only allow human-in-the-loop for web search results
                if source_type == "websearch":
                    print("---WEB SEARCH RESULT: ROUTE TO HUMAN DECISION---")
                    return "useful_websearch"
                else:
                    print("---VECTOR STORE RESULT: END PROCESS---")
                    return "useful_vectorstore"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        

    def route_question_after_attempts(self , state: GraphState):
        """
        Route based on document relevance and attempt count.
        
        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        attempts = state.get("number_of_document_tries", 0)

        if not filtered_documents:
            # No relevant documents found
            print("---NO RELEVANT DOCUMENTS FOUND---")
            print(f"---ATTEMPT NUMBER: {attempts}---")
            
            if attempts >= 3:
                print("---ROUTE TO WEB SEARCH AFTER 3 ATTEMPTS---")
                return "web_search"
            else:
                print("---ROUTE TO TRANSFORM QUERY (CONTINUE TRYING)---")
                return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
        

    def human_in_the_loop(self, state: GraphState):
        """
        Human-in-the-loop intervention for deciding whether to upload web search results.
        
        This node will be interrupted by the graph, allowing human decision making.
        The human can set upload_status in the state to control the next action.

        Args:
            state (dict): The current graph state
            
        Returns:
            state (dict): State with human decision for upload
        """
        print("---HUMAN IN THE LOOP INTERVENTION REQUIRED---")
        print("---WAITING FOR HUMAN DECISION ON UPLOADING WEB SEARCH RESULT---")
        
        question = state["question"]
        generation = state["generation"]
        documents = state["documents"]
        attempts = state.get("number_of_document_tries", 0)
        source_type = state.get("source_type", "unknown")
        
        # The upload_status will be set by human intervention
        # Default to "False" if not set by human
        upload_status = state.get("upload_status", "False")
        
        print(f"Question: {question}")
        print(f"Generated Answer: {generation}")
        print(f"Source Type: {source_type}")
        print(f"Upload Status: {upload_status}")
        
        return {
            "question": question,
            "generation": generation,
            "documents": documents,
            "number_of_document_tries": attempts,
            "upload_status": upload_status,
            "source_type": source_type
        }
        

    def decide_to_upload(self, state: GraphState):
        """
        Decide whether to upload the generated answer to vector store based on human decision.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            str: "yes" to upload, "no" to skip
        """
        print("---DECIDE TO UPLOAD---")
        upload_status = state.get("upload_status", "False")
        
        if upload_status.lower() in ["true", "yes", "1"]:
            print("---DECISION: UPLOAD TO VECTOR STORE---")
            return "yes"
        else:
            print("---DECISION: DO NOT UPLOAD---")
            return "no"
        

    def send_answer_vectorstore(self, state: GraphState):
        """
        Send answer from websearch to vectorstore

        Args:
            state (dict): The current graph state
            
        Returns:
            state (dict): Updated state after upload
        """

        print("---SENDING ANSWER FROM WEBSEARCH TO VECTORSTORE---")
        question = state["question"]
        answer = state["generation"]
        documents = state["documents"]
        attempts = state.get("number_of_document_tries", 0)
        source_type = state.get("source_type", "unknown")

        print("---PREPARING TO UPLOAD GENERATED ANSWER AND SOURCE DOCUMENTS TO ASTRA DB---")
        print(f"---SOURCE TYPE: {source_type}---")

        try:
            uploader = upload_generated_answers(documents, answer)
            upload_result = uploader.upload_answer()
            print("---UPLOAD SUCCESSFUL---")
            
            return {
                "question": question,
                "generation": answer,
                "documents": documents,
                "number_of_document_tries": attempts,
                "upload_status": "completed",
                "source_type": source_type
            }
        except Exception as e:
            print(f"---UPLOAD FAILED: {e}---")
            return {
                "question": question,
                "generation": answer,
                "documents": documents,
                "number_of_document_tries": attempts,
                "upload_status": "failed",
                "source_type": source_type
            }