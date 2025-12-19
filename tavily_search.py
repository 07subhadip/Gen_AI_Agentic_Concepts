import google.generativeai as genai
from tavily import TavilyClient

from config import Config
tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)

search_result = tavily_client.search(query="What is the weather in Berlin?", 
                                     search_depth='advanced',
                                     max_results=3)

print(search_result)

#Namaste Folks, lets start in 2-4 minutes