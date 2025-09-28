from pydantic import BaseModel, Field 
from langchain.prompts import ChatPromptTemplate
from src.llms import groqllm 
from src.llms.groqllm import groqllm




class question_router: 
    def __init__(self):
        self.llm = groqllm().get_llm()

    def get_question_router(self): 
        try :
            # Use JSON mode for better reliability
            system = """You are an expert at routing user questions to either a vectorstore or web search.

            You must respond with ONLY a JSON object in this exact format:
            {{"datasource": "vectorstore"}} or {{"datasource": "web_search"}}

            The vectorstore contains documents about: AI agents, prompt engineering, and adversarial attacks.
            
            Rules:
            - Use "vectorstore" for questions about: AI agents, prompt engineering, adversarial attacks, LLMs, machine learning concepts
            - Use "vectorstore" for simple greetings, casual conversations, or non-specific questions (like "hi", "hello", "how are you")
            - Use "web_search" ONLY for questions requiring current/recent information, news, weather, or specific factual queries not covered by the vectorstore
            
            Default to "vectorstore" unless you're certain the question needs current web information.
            Respond with ONLY the JSON object, nothing else."""
            
            route_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "{question}\n\nResponse (JSON only):"),
            ])

            from langchain_core.output_parsers import JsonOutputParser
            json_parser = JsonOutputParser()
            
            question_router = route_prompt | self.llm | json_parser
            return question_router
        except Exception as e:  
            raise ValueError(f"Error occurred with exception : {e}")

