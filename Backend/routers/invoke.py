from fastapi import APIRouter
from uuid import uuid4
from schemas import GraphResponse, initRequest , resumeRequest
from routers.init import run_graph_and_response

router = APIRouter()



@router.post("/graph/start", response_model=GraphResponse)
async def start_graph(request: initRequest):
    
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "question": request.question,
        "number_of_documents_tries": request.number_of_documents_tries
    }

    return await run_graph_and_response(initial_state, config)