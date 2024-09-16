from langchain_community.embeddings import HuggingFaceEmbeddings
from app.components.vector_db import VectorDB
from app.utilities.settings import settings

class QueryProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(repo_id=settings.embeddings_repo_id)
        self.vector_db = VectorDB()

    def process_query(self, query):
        query_vector = self.embeddings.embed(query)
        results = self.vector_db.search_documents_with_metadata(query_vector)
        return results
