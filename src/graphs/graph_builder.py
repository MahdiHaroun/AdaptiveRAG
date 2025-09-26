from src.nodes.RAG_nodes import RAG_nodes 
from src.states.RAGState import GraphState
from langgraph.graph import StateGraph, START, END
from src.llms.groqllm import groqllm 



class Graph_builder: 
    def __init__(self): 
        self.llm = groqllm().get_llm()
        self.graph = StateGraph(GraphState)


    def build_graph(self):
        """
        Build a graph to generate blogss based on topic

        """
        rag_nodes = RAG_nodes()
        # Define nodes
        web_search = rag_nodes.web_search
        retrieve = rag_nodes.retrieve
        grade_documents = rag_nodes.grade_documents
        generate = rag_nodes.generate 
        transform_query = rag_nodes.transform_query
        route_question = rag_nodes.route_question 
        decide_to_generate = rag_nodes.decide_to_generate
        grade_generation_v_documents_and_question = rag_nodes.grade_generation_v_documents_and_question
        




        

        self.graph.add_node("web_search" , web_search) 
        self.graph.add_node("retrieve" , retrieve)
        self.graph.add_node("grade_documents" , grade_documents)
        self.graph.add_node("generate" , generate)
        self.graph.add_node("transform_query" , transform_query)

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
            decide_to_generate,
            {
                "generate": "generate",
                "transform_query": "transform_query"
            },
        )
        self.graph.add_edge("transform_query", "retrieve")
        self.graph.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported" : END,  # End if hallucinated - avoid infinite loops
                "useful": END,
                "not useful": "transform_query"  # Only retry if answer doesn't address question
            },
        )   

        return self.graph.compile()
    
    def get_compiled_graph(self):
        """
        Get the complete, compiled graph ready for execution
        """
        return self.build_graph()
    

# Create the complete, compiled graph ready for execution
graph = Graph_builder().build_graph()

    







