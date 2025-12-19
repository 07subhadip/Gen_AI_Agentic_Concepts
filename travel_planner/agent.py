from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import  HumanMessage
from tavily import TavilyClient
import config
from typing import TypedDict, Optional


#flagging
try:
    tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
except ImportError:
    print(f"Tavily not installed please install it via `pip install tavily-python`")
    tavily_client = None
except Exception as e:
    print(f"Error initializing Tavily client: {e}")
    tavily_client = None
    
class TravelState(TypedDict):
    destination : Optional[str]
    dates : Optional[str]
    duration : Optional[int]
    budget : Optional[float]
    nationality : Optional[str]
    interests : Optional[str]
    current_question : int
    search_results : dict
    itinerary : Optional[str]
    
    
class TravelAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                          temperature=0.7, 
                                          google_api_key=config.GEMINI_API_KEY,
                                          max_output_tokens=512)
        
    def search_info(self, state: TravelState):
        results = {}
        destination = state['destination']
        dates = state['dates']
        nationality = state['nationality']
        
        if tavily_client:
            try:
                #visa requirements
                visa_query = f"Visa requirements for a {nationality} citizen traveling to {destination}."
                results['visa'] = tavily_client.search(visa_query, max_results=2)
                #weather 
                weather_query = f"Typical weather in {destination} during {dates}."
                results['weather'] = tavily_client.search(weather_query, max_results=2)
                #Restaurants
                restaurant_query = f"Top restaurants in {destination}."
                results['restaurants'] = tavily_client.search(restaurant_query, max_results=2)
                #Travel advisories
                advisory_query = f"Current travel advisories for {destination}. for {nationality} citizens."
                results['advisories'] = tavily_client.search(advisory_query, max_results=2)
            except Exception as e:
                print(f"Error during Tavily search: {e}")
                
        state['search_results'] = results
        return state
    
    def generate_itinerary(self, state: TravelState):
        search_context = ""
        for category, data in state['search_results'].items():
            if data and 'results' in data:
                search_context += f"\n{category.upper()}:\n"
                for result in data['results']:
                    search_context += f"- {result['title']}: {result['content']}\n"
                    
        prompt = f"""
        Act like a senior travel and tour guide and provide a very professional travel itinerary:
        
        Destination: {state['destination']}
        Dates: {state['dates']}
        Duration: {state['duration']} 
        Budget: {state['budget']}
        Nationality: {state['nationality']}
        Interests: {state['interests']}

        Research Info : {search_context}
        
        Create a markdown itenary with:
        1. Visa requirements
        2. Budget breakdown
        3. Day-by-day activities (morning, afternoon, evening)
        4. Recommended restaurants
        5. Useful links to visit
        
        Make it a very practical and a very budget conscious itenary.
        Make sure the details are accurate and up-to-date.
        Avoid generic suggestions and *DO NOT HALLUCINATE*
        Self-Critique your output for any mistakes before finalizing.

        Itenary:
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state['itinerary'] = response.content
        return state
    
    def plan_trip(self, answers: list):
        state = TravelState(
            destination=None,
            dates=None,
            duration=None,
            budget=None,
            nationality=None,
            interests=None,
            current_question=0,
            search_results={},
            itinerary=None
        )
        
