from src.llms.groqllm import groqllm
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class GradeHallucinations(): 
    def __init__(self): 
        self.llm = groqllm().get_llm()

    def get_hallucination_grader(self):
        try: 
            # Use a more robust approach with explicit instructions
            system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

            You must respond with ONLY a JSON object in this exact format:
            {{"binary_score": "yes"}} or {{"binary_score": "no"}}

            Rules:
            - "yes" means the answer is grounded in and supported by the provided facts
            - "no" means the answer contains information not supported by the facts
            - Respond with ONLY the JSON object, nothing else"""
            
            hallucination_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}\n\nResponse (JSON only):"),
            ])

            # Use JSON mode instead of structured output for better reliability
            from langchain_core.output_parsers import JsonOutputParser
            json_parser = JsonOutputParser()
            
            hallucination_grader = hallucination_prompt | self.llm | json_parser
            return hallucination_grader
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")    
