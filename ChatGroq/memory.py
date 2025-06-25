import os
from dotenv import load_dotenv  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore
from langchain_core.runnables.history import RunnableWithMessageHistory  # type: ignore
from langchain_core.runnables import Runnable  # type: ignore
from langchain_community.chat_message_histories import ChatMessageHistory #type: ignore
from langchain.chains import ConversationChain # type: ignore
from langchain_core.chat_history import InMemoryChatMessageHistory #type: ignore



# Load .env
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct", 
    temperature=0.7
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: InMemoryChatMessageHistory(),  
    input_messages_key="input",
    history_messages_key="chat_history"
)

session_id = "chatgroq-session-001"

print("ChatGroq Assistant — type 'exit' to end the chat.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Ending chat. Goodbye!")
        break

    response = with_message_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print("Assistant:", response.content)