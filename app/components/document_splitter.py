import logging
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentSplitter:
    def __init__(self, split_by="paragraph"):
        self.split_by = split_by
        logger.info("Initialized DocumentSplitter with split method '%s'.", self.split_by)

    def split_document(self, document: Document):
        try:
            if self.split_by == "paragraph":
                return self._split_by_paragraph(document)
            elif self.split_by == "page":
                return self._split_by_page(document)
            else:
                raise ValueError("Unsupported split method")
        except Exception as e:
            logger.error("Error splitting document: %s", str(e), exc_info=True)
            raise

    def _split_by_paragraph(self, document: Document):
        try:
            paragraphs = document.page_content.split("\n\n")  # Assuming paragraphs are separated by double newlines
            logger.info("Split document into %d paragraphs.", len(paragraphs))
            return [Document(page_content=para) for para in paragraphs]
        except Exception as e:
            logger.error("Error splitting by paragraph: %s", str(e), exc_info=True)
            raise

    def _split_by_page(self, document: Document):
        try:
            pages = document.page_content.split("\f")  # Assuming form feed (\f) is used for page breaks
            logger.info("Split document into %d pages.", len(pages))
            return [Document(page_content=page) for page in pages]
        except Exception as e:
            logger.error("Error splitting by page: %s", str(e), exc_info=True)
            raise

    def split_documents(self, documents):
        try:
            chunks = []
            for document in documents:
                chunks.extend(self.split_document(document))
            logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))
            return chunks
        except Exception as e:
            logger.error("Error splitting documents: %s", str(e), exc_info=True)
            raise
