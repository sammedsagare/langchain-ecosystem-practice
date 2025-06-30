from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

load_dotenv()

def generate_answers(animal_type):
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7
    )

    prompt_template = PromptTemplate(
        input_variables=['animal_type'],
        template="List three interesting facts about a {animal_type}."
    )

    user_query = prompt_template.format(animal_type=animal_type)

    response = llm.invoke([
        HumanMessage(content=user_query)
    ])

    return response.content

if __name__ == "__main__":
    animal = input("Enter an animal type: ")
    print(generate_answers(animal))
