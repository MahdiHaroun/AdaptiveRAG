from src.llms.groqllm import groqllm
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class question_rewriter:



    def __init__(self): 
        self.llm = groqllm().get_llm()

    def question_rewriter(self): 
        try: 
            system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
            re_write_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                    ),
                ]
            )

            question_rewriter = re_write_prompt | self.llm | StrOutputParser()

            return question_rewriter
        except Exception as e:
            raise ValueError(f"Error occurred with exception : {e}")