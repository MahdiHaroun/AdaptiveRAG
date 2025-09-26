from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv



class embedding: 
    def __init__(self): 
        load_dotenv()

    def get_embedding(self):
        try: 
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return embedding
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")