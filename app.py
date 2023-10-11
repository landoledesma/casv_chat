import streamlit as st 
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from conversation import load_data, process_data, conversational_chat


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



def retrival_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}
    )
    return qa_chain


st.title("Chat con CSV ")


uploaded_file = st.sidebar.file_uploader("Upload your CSV", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    chain = process_data(data)
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hola ! Preguntame " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Habla con tu csv aqui (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(chain, user_input, st.session_state['history'])
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

