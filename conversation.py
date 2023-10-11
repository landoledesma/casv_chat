import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
import openai
from dotenv import load_dotenv

# Cargando variables de entorno
load_dotenv("token.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_llm():
    llm  = ChatOpenAI(
        max_tokens=1000,
        temperature=0.5
        )
    return llm

def conversational_chat(chain,query,history):
    result = chain({"question":query,"chat_history":history})
    return result["answer"]