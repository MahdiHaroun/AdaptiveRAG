from src.llms.groqllm import groqllm
from src.states.RAGState import GradeAnswer
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class answer_grader: 
    def __init__(self): 
        self.llm = groqllm().get_llm()

    def get_answer_grader(self):
        try: 
            # Use JSON mode for better reliability
            system = """You are a grader assessing whether an LLM generation answers a user question.

            You must respond with ONLY a JSON object in this exact format:
            {{"binary_score": "yes"}} or {{"binary_score": "no"}}

            Rules:
            - "yes" means the answer addresses and answers the user's question
            - "no" means the answer does not address the user's question
            - Respond with ONLY the JSON object, nothing else"""
            
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}\n\nResponse (JSON only):"),
            ])

            from langchain_core.output_parsers import JsonOutputParser
            json_parser = JsonOutputParser()
            
            answer_grader = answer_prompt | self.llm | json_parser
            return answer_grader
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")