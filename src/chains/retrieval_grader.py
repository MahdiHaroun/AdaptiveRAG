from pydantic import BaseModel, Field 
from langchain.prompts import ChatPromptTemplate
from src.llms import groqllm 
from src.states.RAGState import GradeDocuments 
from src.llms.groqllm import groqllm



class retrieval_grader: 
    """
    a class to represent retrieval grader
    """

    def __init__(self):
        self.llm = groqllm().get_llm()
        
        

    def get_retrieval_grader(self): 
        try :
            # Use JSON mode for better reliability
            system = """You are a grader assessing relevance of a retrieved document to a user question.

            You must respond with ONLY a JSON object in this exact format:
            {{"binary_score": "yes"}} or {{"binary_score": "no"}}

            Rules:
            - If the document contains keywords or semantic meaning related to the user question, grade it as "yes"
            - It does not need to be a stringent test. The goal is to filter out completely irrelevant retrievals
            - "yes" means the document is relevant to the question
            - "no" means the document is not relevant to the question
            - Respond with ONLY the JSON object, nothing else"""
            
            grade_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}\n\nResponse (JSON only):"),
            ])
            
            from langchain_core.output_parsers import JsonOutputParser
            json_parser = JsonOutputParser()
            
            retrieval_grader = grade_prompt | self.llm | json_parser
            return retrieval_grader
        except Exception as e:  
            raise ValueError(f"Error occurred with exception : {e}")




