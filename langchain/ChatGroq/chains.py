from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

def generate_answers(animal_type, pet_gender):
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7
    )

    prompt_template = PromptTemplate(
        input_variables=['animal_type', 'pet_gender'],
        template="List three unique names for a {pet_gender} {animal_type}."
    )

    chain = RunnableSequence(prompt_template, llm)

    response = chain.invoke({'animal_type': animal_type, 'pet_gender': pet_gender})

    return response.content

if __name__ == "__main__":
    animal = input("Enter the animal type: ")
    gender = input(f"Enter {animal} gender: ")
    print(generate_answers(animal, gender))
