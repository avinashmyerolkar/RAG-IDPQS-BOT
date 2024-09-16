from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from app.utilities.settings import settings

class ModelLoader:
    _embedding_model = None
    _chat_model = None

    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            cls._embedding_model = HuggingFaceEmbeddings()
        return cls._embedding_model

    @classmethod
    def get_chat_model(cls):
        if cls._chat_model is None:
            cls._chat_model = ChatOpenAI(model_name=settings.open_ai_model, 
                                         api_key=settings.open_ai_key, 
                                         temperature=0)
        return cls._chat_model
    


