from fastapi import APIRouter
from uuid import uuid4
from schemas import GraphResponse, initRequest , resumeRequest
from routers.init import run_graph_and_response, get_compiled_graph

router = APIRouter()





@router.post("/graph/resume", response_model=GraphResponse)
async def resume_graph(request: resumeRequest):
    compiled_graph = get_compiled_graph()
    config = {"configurable": {"thread_id": request.thread_id}}
    state = {"upload_status": request.upload_status}
    print(f"State to update: {state}")
    compiled_graph.update_state(config, state)

    return await run_graph_and_response(None, config)