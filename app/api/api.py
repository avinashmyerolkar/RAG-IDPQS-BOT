import os
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.components.document_processor import DocumentProcessor
from app.components.vector_db import VectorDB
from app.components.response_generator import ResponseGenerator
from app.utilities.settings import settings
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format=settings.log_format)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
document_processor = DocumentProcessor(split_by=settings.split_by)
vector_db = VectorDB()
response_generator = ResponseGenerator()

@router.post("/process_documents/")
async def process_documents(files: list[UploadFile] = File(...)):
    saved_files = []
    try:
        for file in files:
            file_location = os.path.join(settings.uploaded_pdfs_path, file.filename)
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())
            saved_files.append(file_location)
            logger.info("Saved file '%s' to '%s'.", file.filename, file_location)

        with ThreadPoolExecutor() as executor:
            futures = []
            for pdf_path in saved_files:
                futures.append(executor.submit(document_processor.load_and_split_pdfs, pdf_path))

            for future in futures:
                document_chunks = future.result()
                key_value_data = document_processor.extract_key_value_pairs(document_chunks)
                vector_db.initialize_vector_store()
                vector_db.store_documents_with_metadata(key_value_data)

        logger.info("Documents processed, key-value pairs extracted, and stored successfully.")
        return {"message": "Documents processed and stored successfully"}

    except Exception as e:
        logger.error("Failed to process documents: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Unable to process documents.")

@router.post("/query/")
async def query_system(query: str = Form(...)):
    logger.info("Received query: %s", query)
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(response_generator.generate_response, query)
            response = future.result()

        logger.info("Response generated successfully.")
        return {"response": response}

    except Exception as e:
        logger.error("Failed to generate response: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Unable to generate response.")
