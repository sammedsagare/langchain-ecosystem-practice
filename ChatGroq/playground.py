import os

from dotenv import load_dotenv # type: ignore
from langchain_groq import ChatGroq # type: ignore
from langchain_core.messages import HumanMessage # type: ignore


# Load .env
load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct", 
    temperature=0.7
)


# user input
user_query = input("Enter your question for the LLM: ")

# send message
response = llm.invoke([
    HumanMessage(content=user_query)
])

os.system('cls' if os.name == 'nt' else 'clear')
print(response.content)
