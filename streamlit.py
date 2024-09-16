import streamlit as st
import requests
from app.utilities.settings import settings

# Define the FastAPI endpoints from settings
PROCESS_DOCUMENTS_URL = settings.process_documents_url
QUERY_URL = settings.query_url

st.title("Intelligent Document Processing and Query System")

# Initialize session state to manage button states
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Upload PDF Files
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

# Process Documents Button
process_button = st.button("Process Documents", disabled=st.session_state.processing)
if process_button:
    st.session_state.processing = True  # Disable the buttons during processing
    if uploaded_files:
        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        with st.spinner('Processing documents...'):
            response = requests.post(PROCESS_DOCUMENTS_URL, files=files)
        if response.status_code == 200:
            st.success("Documents processed, key-value pairs extracted, and stored successfully!")
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please upload at least one PDF file.")
    st.session_state.processing = False  # Re-enable the buttons after processing

# Query Input
query = st.text_input("Enter your query")

# Query Button
query_button = st.button("Submit Query", disabled=st.session_state.processing)
if query_button:
    st.session_state.processing = True  # Disable the buttons during processing
    if query:
        with st.spinner('Generating response...'):
            response = requests.post(QUERY_URL, data={"query": query})
        if response.status_code == 200:
            result = response.json()
            st.write("Response:", result["response"])
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please enter a query.")
    st.session_state.processing = False  # Re-enable the buttons after processing
