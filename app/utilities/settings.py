
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    # Hugging Face API Token and Model Configurations
    huggingfacehub_api_token: str
    embeddings_repo_id: str
    llm_repo_id: str
    open_ai_model: str
    open_ai_key: str

    # ChromaDB Settings
    store_name: str
    persist_directory: str

    # PDF Directory Paths
    pdf_path: str
    uploaded_pdfs_path: str

    # Streamlit URLs
    process_documents_url: str
    query_url: str

    # FastAPI Settings
    host: str
    port: int

    # Logging Settings
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s'

    # Text Splitting Settings
    split_by: str
    chunk_size: int
    chunk_overlap: int

    # Prompt Template Path
    prompt_template_path: str

    # Query Settings
    top_retriever_count: int

    #Token Limit
    token_limit: int

    class Config:
        env_file = ".env"

# Initialize settings
settings = Settings()


#print("pdf_file_path_is :",settings['pdf_path'])

#print("from app.utilities.settings.py",settings.dict())  # This will print all the settings loaded from the .env file
