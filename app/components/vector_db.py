import os
import logging
from langchain_community.vectorstores import Chroma
from app.utilities.settings import settings
from app.utilities.model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.collection_name = settings.store_name
        self.persist_directory = settings.persist_directory
        self.vectorstore = None
        self.embedding_function = ModelLoader.get_embedding_model()
        logger.info("Initialized VectorDB with collection name '%s' and persist directory '%s'.", self.collection_name, self.persist_directory)

    def initialize_vector_store(self):
        try:
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)
                logger.info("Created persist directory '%s'.", self.persist_directory)
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            logger.info("Vector store initialized.")
        except Exception as e:
            logger.error("Failed to initialize vector store: %s", str(e), exc_info=True)
            raise

    def store_documents_with_metadata(self, document_chunks):
        try:
            logger.info(f"Storing documents with metadata")
            self.vectorstore = Chroma.from_documents(documents=document_chunks, embedding=self.embedding_function,
                                                 collection_name=settings.store_name,
                                                 persist_directory=settings.persist_directory)

            self.vectorstore.persist()
            logger.info("Documents and metadata stored and vector store persisted.")
        except Exception as e:
            logger.error("Failed to store documents with metadata: %s", str(e), exc_info=True)
            raise

    def search_documents_with_metadata(self, query):
        try:
            logger.info("Searching for documents with query: %s", query)
            results = self.vectorstore.similarity_search(query)
            documents_with_metadata = []
            for result in results:
                document = result.page_content
                metadata = result.metadata
                documents_with_metadata.append({"document": document, "metadata": metadata})
            logger.info("Search completed, returning documents with metadata.")
            return documents_with_metadata
        except Exception as e:
            logger.error("Failed to search documents with metadata: %s", str(e), exc_info=True)
            raise
