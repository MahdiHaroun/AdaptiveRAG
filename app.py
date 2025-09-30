from fastapi import FastAPI, HTTPException , UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import traceback
import logging
from contextlib import asynccontextmanager
from db_test import test_astra_connection
from utils.documents_uplaoder import PDFChunksUploader


# Import your RAG components
from src.graphs.graph_builder import Graph_builder
from src.states.RAGState import RAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the compiled graph
compiled_graph = None
datastax_status = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global compiled_graph
    global datastax_status
    
    try:
        logger.info("Checking Datastax Astra DB connection...")
        datastax_status = test_astra_connection()
        if datastax_status == True:
            logger.info("Datastax Astra DB connection successful!")
        else:
            logger.error("Datastax Astra DB connection failed!")
            raise ValueError("Cannot connect to Datastax Astra DB")
        logger.info("Initializing RAG system...")
        # Initialize and compile the graph on startup
        graph_builder = Graph_builder()
        compiled_graph = graph_builder.build_graph()
        logger.info("RAG system initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Shutting down RAG system...")

# Create FastAPI app with lifespan events
app = FastAPI(
    title="Adaptive RAG FastAPI",
    description="An adaptive RAG system using LangGraph with question routing, document grading, and generation quality control",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system", min_length=1)
    number_of_documents_attempted: int = Field(..., description="Number of document retrieval attempts")
    
class RAGResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    source_documents: List[str] = Field(default=[], description="Source documents used for the answer")
    question: str = Field(..., description="The original question")
    success: bool = Field(default=True, description="Whether the request was successful")
    
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    success: bool = Field(default=False, description="Whether the request was successful")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Adaptive RAG FastAPI Server",
        "description": "Send POST requests to /ask_rag with your questions",
        "endpoints": {
            "/ask_rag": "POST - Ask a question to the RAG system",
            "/health": "GET - Check system health",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global compiled_graph
    
    if compiled_graph is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "status": "healthy",
        "rag_system": "initialized",
        "datastax_astra_db": "connected" if datastax_status else "not connected",
        "message": "RAG system is ready to process questions"
    }

@app.post("/ask_rag", response_model=RAGResponse)
async def ask_rag(request: QuestionRequest):
    """
    Ask a question to the adaptive RAG system
    
    The system will:
    1. Route the question to either web search or vector store
    2. Retrieve and grade relevant documents  
    3. Generate an answer with quality control
    4. Return the answer with source information
    """
    global compiled_graph
    
    if compiled_graph is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized. Please check server logs."
        )
    
    try:
        logger.info(f"Processing question: {request.question}")
        
        # Prepare the input state for the graph (only question needed, counter will be initialized)
        input_state = {
            "question": request.question.strip()
        }
        
        # Execute the graph with memory support
        logger.info("Executing RAG graph...")
        config = {"configurable": {"thread_id": f"conversation_{hash(request.question) % 10000}"}}
        result = compiled_graph.invoke(input_state, config=config)
        
        # Extract the results
        answer = result.get("generation", "No answer generated")
        documents = result.get("documents", [])
        
        # Process documents to extract source information
        source_docs = []
        if documents:
            if isinstance(documents, list):
                for doc in documents:
                    if hasattr(doc, 'page_content'):
                        # Truncate long documents for response
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        source_docs.append(content)
                    elif isinstance(doc, str):
                        content = doc[:200] + "..." if len(doc) > 200 else doc
                        source_docs.append(content)
            elif hasattr(documents, 'page_content'):
                # Single document
                content = documents.page_content[:200] + "..." if len(documents.page_content) > 200 else documents.page_content
                source_docs.append(content)
            elif isinstance(documents, str):
                content = documents[:200] + "..." if len(documents) > 200 else documents
                source_docs.append(content)
        
        logger.info(f"Generated answer with {len(source_docs)} source documents")
        
        return RAGResponse(
            answer=answer,
            source_documents=source_docs,
            question=request.question,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )


@app.post("/upload-pdf/")
async def upload_pdf(files: list[UploadFile] = File(...)):
    try:
        pdf_files = [(file.filename, await file.read()) for file in files]
        uploader = PDFChunksUploader(
            files=[(filename, memoryview(content)) for filename, content in pdf_files]
        )
        result = uploader.start_pdf_upload()
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
