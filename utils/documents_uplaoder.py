import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.embedding.embedding import embedding
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv

load_dotenv()

class PDFChunksUploader:
    def __init__(self, files):
        self.files = files
        self.embedder = embedding().get_embedding()

    def get_vectorstore(self):
        api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
        token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
        

        if not api_endpoint or not token:
            raise ValueError("Astra DB API endpoint or token not set in environment variables.")

        return AstraDBVectorStore(
            embedding=self.embedder,
            api_endpoint=api_endpoint,
            token=token,
            collection_name="astra_vector_langchain",
        )

    def process_pdf_and_split(self, vectorstore):
        try:
            total_chunks_uploaded = 0
            for filename, file_bytes in self.files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                loader = PyMuPDFLoader(tmp_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                pdf_chunks = text_splitter.split_documents(documents)

                vectorstore.add_documents(pdf_chunks)
                total_chunks_uploaded += len(pdf_chunks)
                os.remove(tmp_path)

            return {
                "message": "Upload and processing complete.",
                "total_chunks_uploaded": total_chunks_uploaded
            }

        except Exception as e:
            raise ValueError(f"Error while processing PDFs: {e}")

    def start_pdf_upload(self):
        vectorstore = self.get_vectorstore()
        return self.process_pdf_and_split(vectorstore)
