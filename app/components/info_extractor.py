import logging
from app.utilities.model_loader import ModelLoader
from app.utilities.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfoExtractor:
    def __init__(self):
        self.embedding_model = ModelLoader.get_embedding_model()
        logger.info("Initialized InfoExtractor with HuggingFaceEmbeddings model.")

    def extract_and_embed(self, documents):
        tagged_documents = []
        try:
            logger.info("Embedding document with content length %d.", len(documents))
            embedded_vector = self.embedding_model.embed_documents([documents])
            tagged_documents.append({"document": documents, "vector": embedded_vector})
            logger.info("Document embedded successfully.")

            logger.info("Processed %d documents.", len(documents))
            return tagged_documents

        except Exception as e:
            logger.error("Error extracting and embedding documents: %s", str(e), exc_info=True)
            raise
