from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from crewai.tools import BaseTool
from pydantic import Field

class DuckDuckGo(BaseTool):
    name: str = "DuckDuck Go Search Tool"
    description: str = "Useful for search based queries. Used primarily to extract insights from the internet"
    search: DuckDuckGoSearchRun = Field(default_factory=DuckDuckGoSearchRun)

    def _run(self,query:str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {e}"
