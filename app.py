import os
import streamlit as st
import pdfplumber
from transformers import pipeline

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def split_text_into_chunks(text, chunk_size=1024):
    """Split text into smaller chunks for processing."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Streamlit UI
st.title("📄 Document Summarizer")

uploaded_file = st.file_uploader("Upload a Document (.txt or .pdf)", type=["txt", "pdf"])
summary_length = st.number_input("Summary Length (max words)", min_value=10, max_value=500, value=100)

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File uploaded successfully: {uploaded_file.name}")
    
    # Extract text based on file type
    if uploaded_file.type == "text/plain":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(file_path)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Summarization
    if text:
        st.subheader("Extracted Text Preview:")
        st.text_area("Original Text", text[:2000], height=150)  # Show first 2000 characters
        
        if st.button("Generate Summary"):
            chunks = split_text_into_chunks(text)
            summary_results = []
            
            for chunk in chunks:
                summary = summarizer(chunk, max_length=summary_length, min_length=20, do_sample=False)
                summary_results.append(summary[0]['summary_text'])

            final_summary = " ".join(summary_results)  # Combine chunk summaries
            st.subheader("Summary:")
            st.write(final_summary)
