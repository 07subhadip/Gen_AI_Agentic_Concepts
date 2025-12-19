import google.generativeai as genai
from config import Config
import re
from langchain.messages import  HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_api_key = Config.GEMINI_API_KEY
genai.configure(api_key=gemini_api_key)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             google_api_key=gemini_api_key,
                             max_output_tokens=1024, 
                             temperature=0.7)


#tool for my agent to do calculations
def calculate(expression):
    return eval(expression)

#tool for my agent to get average food items
def average_food_items(name):
    food_warehouse = {
        "chicken": "100 grams",
        "mutton": "200 grams",
        "rice": "300 grams",
        "wheat": "250 grams",
        "corn": "400 grams"
    }
    return f"The Average food quantity for {name} is {food_warehouse.get(name, 'Item not found')}."

#Available tools for your agent to use
known_actions = {
    "calculate": calculate,
    "average_food_items": average_food_items
}

class Agent:
    def __init__(self):
        self.messages = [] #message history
        
    def __call__(self, message):
        self.messages.append(HumanMessage(content=message))
        response = self.execute()
        self.messages.append(AIMessage(content=response))
        return response
    
    def execute(self):
        response = llm.invoke(self.messages)
        return response.content
    
    
action_re = re.compile(r'Action:\s*(\w+):\s*(.*?)(?:\n|$)',re.MULTILINE)

def run_query(prompt, max_turns=10):
    agent = Agent()
    next_prompt = prompt
    
    for turn in range(max_turns):
        print(f"Turn {turn+1}:")
        response = agent(next_prompt)
        print(f"Assistant: \n{response}\n")
        
        actions = action_re.findall(response)
        
        if actions:
            action, action_input = actions[0]
            action_input = action_input.strip()
            
            if action not in known_actions:
                raise ValueError(f"Unknown action: {action}")
            
            observation = known_actions[action](action_input)
            print(f"Observation: {observation}\n")
            next_prompt = f"Observation: {observation}"
        else:
            if "Answer:" in response:
                print("Final Answer found. Ending interaction.")
                break
            else:
                print("No action found. Ending interaction.")
                break
            
instruction = """
You are a helpful assistant to follow a REACT reasoning pattern.

for each step:
1. Use "Thought:" to think step by step about the problem i am passing to you
2. If you need more information, use "Action : <action_name>: <input>" on a new line 
3. After each action, I will provide you an 'Observation:' with the result of the action.
4. Continue thinking and acting until you can provide a final 'Answer:'

Available actions:
- calculate: to perform mathematical calculations. Input should be a valid mathematical expression.
- average_food_items: to get average food items for a given food name. Input should be the name of the food item.   
Important: Only perform ONE action at a time. then wait for the observation.

Question : I have 2 chickens and 3 muttons. What is the total food quantity I have?
"""

rq = run_query(instruction)
print("Final Response from Agent:")
print(rq)