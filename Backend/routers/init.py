from fastapi import APIRouter, HTTPException
from schemas import GraphResponse

router = APIRouter()

# Import the compiled_graph from main module
def get_compiled_graph():
    from main import compiled_graph
    return compiled_graph

async def run_graph_and_response(input_state, config):
    compiled_graph = get_compiled_graph()
    
    if compiled_graph is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    result = compiled_graph.invoke(input_state, config)
    state = compiled_graph.get_state(config)
    print(state)
    next_nodes = state.next
    thread_id = config["configurable"]["thread_id"]
    if next_nodes and "human_feedback" in next_nodes:
            run_status = "user_feedback"
    else:
            run_status = "finished"
    return GraphResponse(
                thread_id=thread_id,
                run_status=run_status,
                answer=result.get("generation", None),
                number_of_documents_tries=result.get("number_of_documents_tries", 0),
                answer_source=result.get("source_type", None)
            )