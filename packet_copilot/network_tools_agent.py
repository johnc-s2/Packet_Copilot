import os
import json
import time
import logging
import requests
import subprocess
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)


# ---------------------------- CLASS-BASED IP INTELLIGENCE ---------------------------- #
class IPIntelligence:
    """
    Provides multiple intelligence gathering capabilities for public IPs.
    - Reverse DNS lookup (nslookup)
    - Whois lookup (IP ownership & registration info)
    - BGP lookup (ASN & routing details)
    - Threat intelligence (Blacklist & reputation score)
    """

    def __init__(self, ip):
        self.ip = ip

    def run_command(self, command):
        """Runs a shell command and returns the output."""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.stdout else "No data found."
        except subprocess.TimeoutExpired:
            return "Command timed out."
        except Exception as e:
            logging.error(f"Command execution failed: {e}")
            return "Error executing command."

    def nslookup(self):
        """Performs a reverse DNS lookup for the IP address."""
        return self.run_command(f"nslookup {self.ip}")

    def whois_lookup(self):
        """Performs a Whois lookup for the IP address."""
        return self.run_command(f"whois {self.ip}")

    def bgp_lookup(self):
        """Queries BGPView API for ASN and routing information."""
        url = f"https://api.bgpview.io/ip/{self.ip}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"BGP lookup failed: {e}")
            return "Error retrieving BGP data."

    def threat_check(self):
        """Checks if the IP address is blacklisted or has a poor reputation score."""
        url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={self.ip}"
        headers = {"Key": ABUSEIPDB_API_KEY}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Threat intelligence lookup failed: {e}")
            return "Error retrieving threat intelligence data."

    def get_intelligence(self):
        """Aggregates all intelligence into a single response."""
        return {
            "IP": self.ip,
            "Reverse DNS": self.nslookup(),
            "Whois Info": self.whois_lookup(),
            "BGP Data": self.bgp_lookup(),
            "Threat Intelligence": self.threat_check()
        }


# ---------------------------- TOOL FUNCTIONS ---------------------------- #

def parse_input(input_data):
    """Ensures input_data is a dictionary."""
    if isinstance(input_data, dict):
        return input_data
    try:
        return json.loads(input_data) if isinstance(input_data, str) else {"error": "Invalid input format."}
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return {"error": "Invalid JSON format."}


def get_ip_intelligence(input_data):
    """Runs multiple intelligence checks on a public IP address."""
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    if "ip" not in input_data:
        return {"error": "Missing 'ip' field."}

    ip_intel = IPIntelligence(input_data["ip"])
    return ip_intel.get_intelligence()


# ---------------------------- LANGCHAIN TOOLS ---------------------------- #

get_ip_intel_tool = Tool(
    name="fetch_ip_intelligence",
    description="Performs nslookup, whois, BGP lookup, and threat analysis on a public IP.",
    func=get_ip_intelligence
)

# Register tools
tools = [get_ip_intel_tool]

tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.0)

ip_intel_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are an AI Agent that retrieves intelligence on public IP addresses.
    
    You have access to:
    - **fetch_ip_intelligence**: Performs nslookup, whois, BGP lookup, and threat analysis.

    Example:
    - *Question*: What can you tell me about IP 8.8.8.8?
      Thought: I need to gather intelligence on this IP.
      Action: fetch_ip_intelligence
      Action Input: {{ "ip": "8.8.8.8" }}
      Observation: {{
          "Reverse DNS": "dns.google",
          "Whois Info": "Google LLC, United States",
          "BGP Data": "ASN 15169, Route 8.8.8.0/24",
          "Threat Intelligence": "No malicious activity detected."
      }}
      Final Answer: The IP 8.8.8.8 is associated with Google LLC, United States. It has ASN 15169 and belongs to the 8.8.8.0/24 route. No known malicious activity detected.

    **Begin!**
    
    Question: {input}

    {agent_scratchpad}
    """
)

# ---------------------------- CREATE AI AGENT ---------------------------- #

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=ip_intel_prompt.partial(tool_names=tool_names, tools=tool_descriptions)
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
logging.info("🚀 Public IP Intelligence Agent initialized.")


# ---------------------------- TEST EXECUTION ---------------------------- #
if __name__ == "__main__":
    test_input = {"input": {"ip": "8.8.8.8"}}
    response = agent_executor.invoke(test_input)
    print("IP Intelligence Agent Response:", response)