from src.nodes.RAG_nodes import RAG_nodes 
from src.states.RAGState import GraphState
from langgraph.graph import StateGraph, START, END
from src.llms.groqllm import groqllm
from src.embedding.embedding import embedding
from langgraph.checkpoint.memory import MemorySaver



class Graph_builder:    
    def __init__(self): 
        self.llm = groqllm().get_llm()
        self.graph = StateGraph(GraphState)
        self.embedding = embedding().get_embedding()
        self.memory_saver = MemorySaver()


    def build_graph(self):
        """
        Build a graph to generate answers to questions using a RAG approach.
        State is automatically initialized on graph invocation, allowing
        short-term memory to persist within a single execution.
        """
        rag_nodes = RAG_nodes()
        
        # Define nodes
        web_search = rag_nodes.web_search
        retrieve = rag_nodes.retrieve
        grade_documents = rag_nodes.grade_documents
        generate = rag_nodes.generate 
        transform_query = rag_nodes.transform_query
        route_question = rag_nodes.route_question 
        grade_generation_v_documents_and_question = rag_nodes.grade_generation_v_documents_and_question
        route_question_after_attempt = rag_nodes.route_question_after_attempts
        human_in_the_loop = rag_nodes.human_in_the_loop
        send_answer_vectorstore = rag_nodes.send_answer_vectorstore
        decide_to_upload = rag_nodes.decide_to_upload

        # Add nodes to graph
        self.graph.add_node("web_search", web_search) 
        self.graph.add_node("retrieve", retrieve)
        self.graph.add_node("grade_documents", grade_documents)
        self.graph.add_node("generate", generate)
        self.graph.add_node("transform_query", transform_query)
        self.graph.add_node("human_in_the_loop", human_in_the_loop)
        self.graph.add_node("send_answer_vectorstore", send_answer_vectorstore)

        # Start directly with routing the question (no initialization node needed)
        self.graph.add_conditional_edges(
            START, 
            route_question, 
            {
                "web_search": "web_search",
                "vectorstore": "retrieve"
            },
        )
        
        self.graph.add_edge("web_search", "generate")
        self.graph.add_edge("retrieve", "grade_documents")
        
        self.graph.add_conditional_edges(
            "grade_documents",
            route_question_after_attempt,
            {
                "generate": "generate", 
                "web_search": "web_search",
                "transform_query": "transform_query"
            },
        )
        
        self.graph.add_edge("transform_query", "retrieve")

        self.graph.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": END,  # End if hallucinated - avoid infinite loops
                "useful_websearch": "human_in_the_loop",  # Go to human decision for web search uploads
                "useful_vectorstore": END,  # End directly for vector store results
                "not useful": "transform_query"  # Only retry if answer doesn't address question
            },
        )   

        # Human decides whether to upload the web search result to vector store
        self.graph.add_conditional_edges(
            "human_in_the_loop",
            decide_to_upload,
            {
                "yes": "send_answer_vectorstore",
                "no": "transform_query"
            },
        )
        
        self.graph.add_edge("send_answer_vectorstore", END)

        return self.graph.compile(
            interrupt_before=["human_in_the_loop"],
            checkpointer=self.memory_saver
        )
    
    def get_compiled_graph(self):
        """
        Get the complete, compiled graph ready for execution
        """
        return self.build_graph()


# Create the complete, compiled graph ready for execution
graph = Graph_builder().build_graph()