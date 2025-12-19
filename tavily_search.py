import google.generativeai as genai
from tavily import TavilyClient

from config import Config
tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)

search_result = tavily_client.search(query="who is the prime minister of india", 
                                     search_depth='advanced',
                                     max_results=4)

for result in search_result.get('results', []):
    print(f"Title: {result['title']}")
    print("=======")
    print(f"Content: {result['content']}\n")
    print("=======")
    print(f"URL: {result['url']}\n")