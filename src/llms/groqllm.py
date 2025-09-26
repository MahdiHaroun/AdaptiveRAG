from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os 



class groqllm: 
    def __init__(self):
        load_dotenv()

    def get_llm(self ):
        try: 
            os.environ["GROQ_API_KEY"]=self.groq_api_key=os.getenv("GROQ_API_KEY")
            # Use Llama 3.3 70B - supports tool use, parallel tool use, and JSON mode
            llm=ChatGroq(
                model="moonshotai/kimi-k2-instruct-0905",  # Latest model with full tool support
                temperature=0,
                max_tokens=1024
            )
            return llm
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")


        
