from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

video_url = "https://youtu.be/IFNkhFEpWNI?si=es1C_pa5RbgIQuFh"

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

# print(create_vector_db_from_youtube_url(video_url))

def get_response_from_query(db, query, k=4):
    # text-davinci can handle until 4097 tokens

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="gpt-3.5-turbo")

    prompt = PromptTemplate.from_template(
        # input_variables = ["question", "docs"],
        template = """
        You are a helpful YouYube assistant that can answer questions about videos based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be detailed.
        """
    )

    chain = LLMChain(llm = llm, prompt = prompt)

    response = chain.run(question = query, docs = docs_page_content)
    response = response.replace("\n", "")
    return response