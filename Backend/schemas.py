from pydantic import BaseModel
from typing import Optional , Literal 


class initRequest(BaseModel):
    question: str 
    number_of_documents_tries: int = 0
    
    
class resumeRequest(BaseModel):
    thread_id: str
    upload_status: Literal["yes", "no"]

class GraphResponse(BaseModel):
    thread_id: str
    run_status: Literal["finished", "human_in_the_loop"]
    answer : Optional[str] = None
    number_of_documents_tries : int
    answer_source : Optional[str] = None
    
