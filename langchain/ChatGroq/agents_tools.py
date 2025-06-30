from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv()

def lang_agent(query):
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7
    )
    
    tools = load_tools(['wikipedia'], llm=llm)
    
    agent =  initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    result = agent.invoke(query)  

    return result['output']  

if __name__ == "__main__":
    user_query = input("Enter your query: ") 
    print(lang_agent(user_query))
