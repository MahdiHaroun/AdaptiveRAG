from fastapi import FastAPI , HTTPException 
from fastapi.middleware.cors import CORSMiddleware

import traceback
import logging
import sys
import os
from contextlib import asynccontextmanager

# Add parent directory to Python path to access src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_test import test_astra_connection
from src.graphs.graph_builder import Graph_builder
from routers import invoke , resume , init 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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



app = FastAPI(
    title="Adaptive RAG FastAPI",
    description="An adaptive RAG system using LangGraph with question routing, document grading, and generation quality control",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(invoke.router)
app.include_router(resume.router)
app.include_router(init.router)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global compiled_graph
    global datastax_status
    
    if compiled_graph is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    if datastax_status is None or datastax_status == False:
        raise HTTPException(status_code=503, detail="Datastax Astra DB not connected")
    
    return {
       "message": "RAG system is online , Datastax Astra DB connection is online",
    }