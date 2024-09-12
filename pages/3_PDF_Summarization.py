import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to read PDF file
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Function to summarize text using OpenAI API
def summarize_text(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# Function to estimate token cost
def estimate_token_cost(text_length):
    # Estimate number of tokens (1 token ~ 4 characters in English text)
    num_tokens = text_length // 4
    cost_per_token = 0.00006  # Cost per token for text-davinci-003 (as of Aug 2024)
    estimated_cost = num_tokens * cost_per_token
    return estimated_cost

st.title("PDF Summarizer using OpenAI API")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Reading PDF..."):
        text = read_pdf(uploaded_file)
    
    st.write("**Extracted Text:**")
    st.write(text[:1000] + "...")  # Display first 1000 characters as a preview

    with st.spinner("Summarizing..."):
        summary = summarize_text(text)
    
    st.write("**Summary:**")
    st.write(summary)

    # Estimate and display token cost
    estimated_cost = estimate_token_cost(len(text))
    st.write(f"**Estimated Token Cost:** ${estimated_cost:.4f}")

    # Additional cost estimate for a 10-page PDF (assuming similar content density)
    avg_cost_per_page = estimated_cost / (uploaded_file.getbuffer().nbytes / 1024 / 1024) * 10
    st.write(f"**Estimated Cost for a 10-page PDF:** ${avg_cost_per_page:.4f}")

