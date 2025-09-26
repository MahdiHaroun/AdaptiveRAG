from langchain_community.tools.tavily_search import TavilySearchResults

class web_search_tool: 
    def __init__(self): 
        self.tool = TavilySearchResults(k=3)
        
    def get_web_search_tool(self): 
        try: 
            web_search_tool = self.tool
            return web_search_tool
        except Exception as e: 
            raise ValueError(f"Error occurred with exception : {e}")