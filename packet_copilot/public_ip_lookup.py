import os
import json
import time
import logging
import requests
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_API_URL = "https://api.weatherapi.com/v1"

# ---------------------------- API CLASS ---------------------------- #
class IP_API:
    """
    Fetches geographic location based on a public IP address.
    """
    def __init__(self):
        self.api_key = WEATHER_API_KEY

    def fetch_data(self, endpoint, params):
        """
        Generic function to query WeatherAPI endpoints.
        """
        url = f"{BASE_API_URL}/{endpoint}.json"
        params["key"] = self.api_key

        retries = 3  # Retry mechanism
        for attempt in range(retries):
            try:
                logging.debug(f"Fetching data from {url} with params {params}")
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed (Attempt {attempt+1}): {e}")
                time.sleep(2)

        return {"error": f"Failed to fetch data from {endpoint} after multiple attempts."}

    def get_ip_info(self, ip):
        """
        Fetches IP geolocation info from WeatherAPI.
        """
        return self.fetch_data("ip", {"q": ip})

# ---------------------------- TOOL FUNCTIONS ---------------------------- #
def parse_input(input_data):
    """
    Ensures input_data is a dictionary.
    """
    if isinstance(input_data, dict):
        return input_data
    try:
        return json.loads(input_data) if isinstance(input_data, str) else {"error": "Invalid input format."}
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return {"error": "Invalid JSON format."}

def get_ip_info(input_data):
    """
    Fetches public IP address information.
    """
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    if "ip" not in input_data:
        return {"error": "Missing 'ip' field."}

    ip_client = IP_API()
    return ip_client.get_ip_info(input_data["ip"])

# ---------------------------- LANGCHAIN TOOLS ---------------------------- #
get_ip_tool = Tool(
    name="fetch_ip",
    description="Fetches geographic information for a public IP address.",
    func=get_ip_info
)

# Register tool
tools = [get_ip_tool]

tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.2)
ip_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are an AI Agent that retrieves geographic information about public IP addresses.
    
    You have access to:
    - **fetch_ip**: Retrieves geolocation data for a public IP address.

    Example:
    - *Question*: Where is IP 8.8.8.8 located?
      Thought: I need to look up this IP.
      Action: fetch_ip
      Action Input: {{ "ip": "8.8.8.8" }}
      Observation: {{ "country": "United States", "region": "California", "city": "Mountain View", "lat": 37.3861, "lon": -122.0839 }}
      Final Answer: The IP 8.8.8.8 is located in Mountain View, California, USA.

    **Begin!**
    
    Question: {input}

    {agent_scratchpad}
    """
)

# Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=ip_prompt.partial(tool_names=tool_names, tools=tool_descriptions)
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=5,
    max_execution_time=30
)

# Log initialization
logging.info("🚀 Public IP Agent initialized.")
# ---------------------------- TEST EXECUTION ---------------------------- #
if __name__ == "__main__":
    test_input = {"input": {"ip": "8.8.8.8"}}
    response = agent_executor.invoke(test_input)
    print("IP Agent Response:", response)