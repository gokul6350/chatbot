import streamlit as st
import PyPDF2
from PyPDF2 import PdfWriter, PdfReader
import openai
import io
import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Set OpenAI API key
openai.api_key = 'your_openai_api_key_here'

# Log file path
log_file_path = "token_usage_log.txt"

# Function to log token usage
def log_token_usage(prompt_tokens, completion_tokens, model_name):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tokens = prompt_tokens + completion_tokens

    # Assuming the pricing for gpt-4o-mini
    cost_per_token_input = 0.00015  # Cost per input token for GPT-4o-mini
    cost_per_token_output = 0.0006  # Cost per output token for GPT-4o-mini
    total_cost = (prompt_tokens * cost_per_token_input) + (completion_tokens * cost_per_token_output)

    log_message = f"[{current_time}] Model: {model_name}, Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Total Tokens: {total_tokens}, Cost: ${total_cost:.4f}\n"
    
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to count tokens
def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to divide text into chunks
def chunk_text(text, max_tokens=2000):
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in text.split('. '):
        sentence_tokens = count_tokens(sentence + '. ')
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + '. '
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to determine which sentences to annotate using OpenAI
def determine_annotations(text, annotation_level):
    prompt = f"Annotate the following text at a '{annotation_level}' level. Return the exact sentences to annotate (case and space-sensitive):\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that annotates text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# Function to annotate PDF by marking the identified sentences
def annotate_pdf(pdf_file, sentences_to_annotate):
    pdf_reader = PdfReader(pdf_file)
    pdf_writer = PdfWriter()

    for page_num, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        for sentence in sentences_to_annotate.split('\n'):
            if sentence in content:
                # Replace the sentence with an annotated version (e.g., highlight)
                content = content.replace(sentence, f"**{sentence}**")

        # Write the annotated content back to a PDF page
        pdf_writer.add_page(page)

    # Create an in-memory PDF file for download
    output = io.BytesIO()
    pdf_writer.write(output)
    output.seek(0)
    return output

def save_annotated_pdf(pdf_data, original_filename):
    # Create a directory to store annotated PDFs if it doesn't exist
    save_dir = "annotated_pdfs"
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(original_filename)[0]
    save_filename = f"{base_filename}_annotated_{timestamp}.pdf"
    save_path = os.path.join(save_dir, save_filename)

    # Save the PDF
    with open(save_path, "wb") as f:
        f.write(pdf_data.getvalue())

    return save_path

def save_extracted_text(text, original_filename):
    save_dir = "converted_text"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(original_filename)[0]
    save_filename = f"{base_filename}_extracted_{timestamp}.txt"
    save_path = os.path.join(save_dir, save_filename)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return save_path

def save_annotated_response(response_text, original_filename):
    save_dir = "annotated_responses"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(original_filename)[0]
    save_filename = f"{base_filename}_annotated_{timestamp}.txt"
    save_path = os.path.join(save_dir, save_filename)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response_text)
    
    return save_path

st.title("PDF Annotation Tool using OpenAI")

# Step 1: Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        extracted_text_path = save_extracted_text(text, uploaded_file.name)
        st.success(f"Extracted text saved to: {extracted_text_path}")
    
    # Step 2: Choose annotation level
    st.write("Select the annotation level:")
    annotation_level = st.radio("Annotation Level", ("low", "medium", "high"))

    # Step 3: Determine annotations
    if st.button("Annotate PDF"):
        with st.spinner("Determining annotations..."):
            sentences_to_annotate = determine_annotations(text, annotation_level)
            annotated_response_path = save_annotated_response(sentences_to_annotate, uploaded_file.name)
            st.success(f"Annotated response saved to: {annotated_response_path}")
            
            st.write("**Sentences to Annotate:**")
            st.write(sentences_to_annotate)

        with st.spinner("Annotating PDF..."):
            annotated_pdf = annotate_pdf(uploaded_file, sentences_to_annotate)

            # Save the PDF physically
            save_path = save_annotated_pdf(annotated_pdf, uploaded_file.name)
            st.success(f"Annotated PDF saved to: {save_path}")

            # Step 4: Download the annotated PDF
            st.download_button(
                label="Download Annotated PDF",
                data=annotated_pdf,
                file_name=os.path.basename(save_path),
                mime="application/pdf"
            )

# Display the logs in the Streamlit app
st.write("### Token Usage Logs")
if st.button("Show Logs"):
    with open(log_file_path, "r") as log_file:
        logs = log_file.read()
        st.text_area("Logs", logs, height=200)

# Display the content of saved files
if st.button("Show Extracted Text"):
    if 'extracted_text_path' in locals():
        with open(extracted_text_path, "r", encoding="utf-8") as f:
            st.text_area("Extracted Text", f.read(), height=200)
    else:
        st.warning("No extracted text available. Please upload a PDF first.")

if st.button("Show Annotated Response"):
    if 'annotated_response_path' in locals():
        with open(annotated_response_path, "r", encoding="utf-8") as f:
            st.text_area("Annotated Response", f.read(), height=200)
    else:
        st.warning("No annotated response available. Please annotate the PDF first.")
