#!/bin/bash

# Adaptive RAG FastAPI Server Startup Script

echo "🚀 Starting Adaptive RAG FastAPI Server..."
echo "=================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo "GROQ_API_KEY=your_groq_api_key_here"
    echo "TAVILY_API_KEY=your_tavily_api_key_here"
    echo ""
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected!"
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "📦 Installing/checking dependencies..."
uv pip install -e .

echo ""
echo "🌐 Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload