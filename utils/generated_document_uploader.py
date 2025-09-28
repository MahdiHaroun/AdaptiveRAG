from db_test import test_astra_connection
from langchain_astradb import AstraDBVectorStore
from src.embedding.embedding import embedding
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

class upload_generated_answers:

    def __init__ (self, documents, answer): 
        self.source_documents = documents
        self.answer = answer
        self.embedder = embedding().get_embedding()
        self.vectorstore = None


    def upload_answer(self):
        try: 
            api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
            token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')

            testing_db = test_astra_connection()
            if testing_db == True : 
                print("Connected to Astra DB successfully!")
                self.vectorstore = AstraDBVectorStore(
                embedding=self.embedder,
                api_endpoint=api_endpoint,
                token=token,
                collection_name="astra_vector_langchain",
            )
                # Extract text content from documents for metadata (JSON serializable)
                if hasattr(self.source_documents, 'page_content'):
                    # Single document
                    source_text = [self.source_documents.page_content]
                elif isinstance(self.source_documents, list):
                    # List of documents
                    source_text = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in self.source_documents]
                else:
                    # Fallback
                    source_text = [str(self.source_documents)]
                
                Document_to_upload = Document(
                    page_content=self.answer,
                    metadata={"source_documents": source_text, "producer" : "Generated_Web_Search_Answer"}
                )

            print("Uploading answer and source documents to Astra DB...")
            self.vectorstore.add_documents([Document_to_upload])

            return "Upload successful"
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")
