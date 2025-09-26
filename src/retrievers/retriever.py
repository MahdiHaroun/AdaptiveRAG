from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from src.embedding.embedding import embedding 
from langchain_astradb import AstraDBVectorStore
import os 


class retriever: 
    def __init__(self):
        self.embedder = embedding().get_embedding()


    def get_retriever(self ): 

        try : 
            vectorstore = AstraDBVectorStore(
                embedding=self.embedder,
                api_endpoint=os.getenv('ASTRA_DB_API_ENDPOINT'),
                token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'),
                namespace= None,
                collection_name="astra_vector_langchain",
            )

            retriever=vectorstore.as_retriever()
            return retriever
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")
    








    
