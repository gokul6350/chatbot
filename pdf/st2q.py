import streamlit as st
import base64
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from fpdf import FPDF

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the layout
st.set_page_config(layout="wide")

# Function to display PDF
def display_pdf(file):
    bytes_data = file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to extract text from the PDF
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to process the PDF
def process_pdf(pdf):
    text = extract_text_from_pdf(pdf)
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Function to get conversation chain
def get_conversation_chain(vectorstore):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    if the user is asking to annotate the pdf, return the key sentences from the pdf which are important and relevant to the user's question
    in a json format so that the UI can annotate the pdf the key sentences must be case sensitive and must be exact same 
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get response from the conversation chain
def get_response_from_chain(chain, vectorstore, question):
    docs = vectorstore.similarity_search(question)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response

# Function to create a PDF from text
def create_pdf_from_text(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return pdf

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
        tabs = st.tabs(["Original PDF", "Raw Text"])
        
        with tabs[0]:
            display_pdf(uploaded_file)
        
        with tabs[1]:
            raw_text = extract_text_from_pdf(uploaded_file)
            st.text_area("Raw Text", raw_text, height=600)
            if st.button("Download Raw Text as PDF"):
                pdf = create_pdf_from_text(raw_text)
                pdf_output = pdf.output(dest='S').encode('latin1')
                b64 = base64.b64encode(pdf_output).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="raw_text.pdf">Download PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("Upload a document to view.")

with col2:
    st.header("Chat with PDF")
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_file is not None:
        if st.session_state.conversation is None:
            vectorstore = process_pdf(uploaded_file)
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore)

        # Display chat messages from history on app rerun
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about the document:")
        if user_question:
            # Display user message in chat message container
            with chat_container:
                st.chat_message("user").markdown(user_question)
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            response = get_response_from_chain(st.session_state.conversation, st.session_state.vectorstore, user_question)
            ai_response = response['output_text']

            # Display assistant response in chat message container
            with chat_container:
                st.chat_message("assistant").markdown(ai_response)
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

            # Rerun the app to update the chat history display
            st.experimental_rerun()
    else:
        st.write("Please upload a PDF document to start chatting.")

    # Footer
    st.write("The AI can answer questions based on the content of the uploaded PDF.")
