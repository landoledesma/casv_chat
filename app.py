from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv

# Cargando variables de entorno
load_dotenv("token.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

# Definiendo la plantilla para el bot
template_bot = """
            usa la siguiente pieza de información para responder la pregunta de usuario
            si no sabes la respuesta solo di : no tengo ese conocimiento,
            no trates de responderla
            Context:{context}
            Question:{question}
            solo regresa la información relevante y nada más
"""

def custom_prompt():
    """
        Prompt template for QA retrival for each vector store
    """
    prompt = PromptTemplate(template=template_bot, input_variables=["context", "question"])
    return prompt

def load_llm():
    llm  = ChatOpenAI(
        max_tokens=1000,
        temperature=0.5
        )
    return llm

def retrival_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = custom_prompt()
    qa = retrival_qa_chain(llm, qa_prompt, db)
    return qa
