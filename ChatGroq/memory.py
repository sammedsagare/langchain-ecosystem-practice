import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory 
from langchain_core.chat_history import InMemoryChatMessageHistory


load_dotenv()  

sessions = {}  

def get_history(session_id):
    if session_id not in sessions:
        sessions[session_id] = InMemoryChatMessageHistory()
    return sessions[session_id] 

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.7
)

chain = RunnableWithMessageHistory(llm, get_history)

if __name__ == "__main__":
    session_id = "user1"  
    print("Start a conversation. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = chain.invoke(
            user_input,
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()}")
