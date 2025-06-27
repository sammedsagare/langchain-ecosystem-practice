from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

load_dotenv()
llm = ChatGroq(temperature=0.7, model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
client = Client()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

chain = prompt | llm

user_input = input("Enter your prompt: ")

response = chain.invoke({"input": user_input})

print(response.content)


"""
langsmith is used here for tracing and monitoring langchain runs.
by initializing the Client(), we enable logging of our chains, prompts, and responses.
helps us analyze, debug, and optimize llm applications via the langsmith dashboard.
set the required env variables (LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT).
can trace this in langsmith -> tracing projects
"""