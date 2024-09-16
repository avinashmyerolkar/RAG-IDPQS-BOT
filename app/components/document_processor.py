import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from app.utilities.settings import settings
from app.components.response_generator import ResponseGenerator
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, split_by):
        self.pdf_path = settings.pdf_path
        self.text_splitter = CharacterTextSplitter()
        self.response_generator = ResponseGenerator()
        logger.info("Initialized DocumentProcessor with PDF path '%s'.", self.pdf_path)

    def load_and_split_pdfs(self, pdf_file_path=None):
        try:
            loader = DirectoryLoader(self.pdf_path)
            documents = loader.load()
            logger.info("Loaded %d documents from directory '%s'.", len(documents), self.pdf_path)

            if not documents:
                logger.warning("No documents found in directory '%s'.", self.pdf_path)

            document_chunks = self.text_splitter.split_documents(documents)
            logger.info("Split documents into %d chunks.", len(document_chunks))

            return document_chunks

        except Exception as e:
            logger.error("Error loading or splitting documents: %s", str(e), exc_info=True)
            raise

    def extract_key_value_pairs(self, document_chunks):
        try:
            logger.info(f"Extracting key-value pairs from document chunks")
            extracted_data = []

            for chunk in document_chunks:
                chunk_metadata = chunk.metadata

                # Generate key-value pairs using GPT
                key_value_pairs = self.response_generator.generate_key_value_pairs(chunk.page_content)

                # Update the existing metadata with key-value pairs
                chunk_metadata["key_value_pairs"] = key_value_pairs

                # Store the document chunk with its updated metadata
                extracted_data.append(Document(page_content=chunk.page_content, metadata=chunk_metadata))

            logger.info("Key-value pairs extracted successfully.")
            return extracted_data

        except Exception as e:
            logger.error("Error extracting key-value pairs: %s", str(e), exc_info=True)
            raise
