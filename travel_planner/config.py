import os
from dotenv import load_dotenv

# load_dotenv('../.env')
# load_dotenv('.env')
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

QUESTIONS = [
    "Where do you want to go and when?",
    "How long is your trip (in days)?",
    "What's your total budget for the trip (excluding flights)?",
    "What is your passport country?",
    "What are your travel interests (e.g., adventure, relaxation, culture, food)?",
]