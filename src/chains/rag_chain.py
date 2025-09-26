from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from src.llms.groqllm import groqllm 

class rag_chain: 
    def __init__(self):
        self.llm = groqllm().get_llm()
        

    def get_rag_chain(self ):

        try : 
            # Try to use the hub prompt, but fallback to custom prompt
            try:
                prompt = hub.pull("rlm/rag-prompt") 
            except:
                # Fallback to custom prompt
                from langchain.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful AI assistant. Use the provided context to answer questions accurately.
                    
                    If the question is a simple greeting or casual conversation (like "hi", "hello", "how are you"), 
                    respond naturally and friendly without requiring specific context.
                    
                    For knowledge questions, use the provided context to give accurate, helpful answers.
                    If you don't have enough context to answer a question, say so clearly.
                    
                    Context: {context}"""),
                    ("human", "{question}")
                ])
            
            rag_chain = prompt | self.llm | StrOutputParser() 

            return rag_chain
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")

    