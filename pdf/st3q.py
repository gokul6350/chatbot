import streamlit as st
import base64
import io
import os
from dotenv import load_dotenv
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setting up the layout
st.set_page_config(layout="wide")

# Function to display PDF
def display_pdf(file):
    bytes_data = file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to process the PDF
def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Function to get conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Sidebar
st.sidebar.title("GOKUL")
st.sidebar.button("Home")
st.sidebar.button("Search")
st.sidebar.button("Graph")

st.sidebar.title("pdf1py")
uploaded_file = st.sidebar.file_uploader("Upload Document", type=["pdf"])

st.sidebar.title("Prompts to get you started")
st.sidebar.write("Welcome Guide")
st.sidebar.write("Keyboard shortcuts")

# Main content area with split screen
col1, col2 = st.columns(2)

with col1:
    st.header("Document Viewer")
    if uploaded_file is not None:
        st.write(f"**Viewing:** {uploaded_file.name}")
        display_pdf(uploaded_file)
    else:
        st.write("Upload a document to view.")

with col2:
    st.header("Chat with PDF")
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    if uploaded_file is not None:
        if st.session_state.conversation is None:
            vectorstore = process_pdf(uploaded_file)
            st.session_state.conversation = get_conversation_chain(vectorstore)

        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write("Human: ", message.content)
                else:
                    st.write("AI: ", message.content)
    else:
        st.write("Please upload a PDF document to start chatting.")

    # Footer
    st.write("The AI can answer questions based on the content of the uploaded PDF.")
