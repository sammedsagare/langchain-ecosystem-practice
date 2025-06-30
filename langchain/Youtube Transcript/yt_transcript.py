from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def vector_db_yt_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=7):
    # 8192 tokens for our model gemma2-9b-it from ChatGroq Production
    
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = ChatGroq(
        model = "gemma2-9b-it",
        temperature= 0.7
    )
    
    prompt = PromptTemplate(
        input_variables=["question", docs],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos based on the video's transcript.
        
        Answer the following question: {question}
        By searching the video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    
    chain = RunnableSequence(prompt, llm)
    
    response = chain.invoke({'question': query, 'docs': docs_page_content})
    response_text = response.content.replace("\n", "")  

    return response_text, docs

