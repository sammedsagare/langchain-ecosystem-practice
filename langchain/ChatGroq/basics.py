from dotenv import load_dotenv 
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage 


load_dotenv()

def generate_answers():

    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct", 
        temperature=0.7
    )


    user_query = input("Enter your question for the LLM: ")

    response = llm.invoke([
        HumanMessage(content=user_query)
    ])

    return response.content
    
if __name__ == "__main__":
    print(generate_answers())

