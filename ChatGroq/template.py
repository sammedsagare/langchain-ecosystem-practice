import os
from dotenv import load_dotenv  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from langchain.prompts import ChatPromptTemplate  # type: ignore
from langchain.chains import LLMChain  # type: ignore

def main():
    load_dotenv()

    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7
    )

    topic = input("Enter your topic for the LLM: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    #template creation
    prompt = ChatPromptTemplate.from_template("Explain {topic} in 15 words.")

    # llm chain
    chain = prompt | llm
    output = chain.invoke({"topic": topic})

    print(output.content)

if __name__ == "__main__":
    main()