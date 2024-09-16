import yaml
import logging
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from app.utilities.settings import settings
from langchain.chat_models import ChatOpenAI
from app.utilities.model_loader import ModelLoader
from transformers import GPT2Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format=settings.log_format)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    def __init__(self, template_file=settings.prompt_template_path):
        self.prompt_template = self.load_prompt_template(template_file)
        self.endpoint = ModelLoader.get_chat_model()
        self.vector_db = None
        self.Open_AI = ChatOpenAI(model_name=settings.open_ai_model, api_key=settings.open_ai_key, temperature=0)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logger.info("Initialized ResponseGenerator with prompt template from '%s'.", template_file)

    def load_prompt_template(self, filepath):
        try:
            with open(filepath, 'r') as file:
                template_data = yaml.safe_load(file)
            logger.info("Loaded prompt template successfully from '%s'.", filepath)
            return template_data['prompt_template']
        except Exception as e:
            logger.error("Error loading prompt template: %s", str(e), exc_info=True)
            raise

    def set_vector_db(self, vector_db):
        self.vector_db = vector_db
        logger.info("VectorDB set in ResponseGenerator.")

    def generate_response(self, query):
        if not query:
            logger.warning("Received empty query.")
            raise ValueError("Query parameter is required and cannot be empty.")

        try:
            logger.info("Generating response for query: %s", query)

            prompt_template = """
            You are an expert assistant tasked with providing highly accurate responses based strictly on the provided documents and their associated metadata. Your primary goal is to ensure that all answers are directly supported by the information in the documents. Follow these guidelines:

            1. **Strict Adherence to Content**: Only provide information that is explicitly mentioned in the documents or their metadata. Do not infer, assume, or create any information that is not directly supported by the text.

            2. **Handle Out-of-Context Questions**: If the question cannot be answered based on the provided documents or metadata, clearly state: "The information is not available in the provided documents."

            3. **Cite Specific Sections**: Wherever possible, reference the exact section, page, or paragraph where the information was found to support your answer.

            4. **No Fabrication**: Do not generate any information that is not explicitly provided in the documents. If the document does not contain the requested information, simply state that it is "not found."

            **Question**: {question}
            **Context**: 
            {context}

            **Answer**:
            """

            connect_vectorstore = Chroma(
                collection_name=settings.store_name,
                embedding_function=ModelLoader.get_embedding_model(),
                persist_directory=settings.persist_directory
            )
            documents_with_metadata = connect_vectorstore.as_retriever(search_kwargs={'k': settings.top_retriever_count})

            if not documents_with_metadata:
                return "No relevant documents found."

            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            memory_stored = ConversationBufferMemory(memory_key="history", input_key="question")
            qa_chain = RetrievalQA.from_chain_type(llm=self.Open_AI,
                                                   chain_type="stuff",
                                                   retriever=documents_with_metadata,
                                                   chain_type_kwargs={"prompt": prompt,
                                                                      "memory": memory_stored},
                                                   return_source_documents=False)

            query_response = qa_chain.run(query)
            logger.info("Response generated successfully.")
            return query_response

        except Exception as e:
            logger.error("Error in generate_response function: %s", str(e), exc_info=True)
            return f"Error in generate_response function: {e}"

    def generate_key_value_pairs(self, ocr_text):
        try:
            logger.info("Generating key-value pairs from OCR text.")

            prompt_local = """
                You are an expert in extracting and tagging information from FAQ documents. Your task is to extract specific entities from each section of the document. 
                The document is divided into multiple sections, and you need to identify and extract the following information from each section:
                - Equipment name
                - Domain (e.g., electronics, mechanical, software)
                - Model numbers
                - Manufacturer

                Each section should be processed separately. For each section, create a dictionary with the following structure:
                {{
                    "section_title": "<Title of the Section>",
                    "equipment_name": "<Extracted Equipment Name>",
                    "domain": "<Extracted Domain>",
                    "model_numbers": "<Extracted Model Numbers>",
                    "manufacturer": "<Extracted Manufacturer>"
                    }}
                }}

                If a piece of information is not found in a section, use `null` or an empty string as the value for that field.

                STRICTLY respond ONLY with json-dictionary format and nothing else. 
                DO NOT ADD '''json in the beginning of the response.
                Final response must NOT be a string but a dict data type

                Question: {question}
                Context:  {context}

                Answer:
            """
            connect_vectorstore = Chroma(
                collection_name=settings.store_name,
                embedding_function=ModelLoader.get_embedding_model(),
                persist_directory=settings.persist_directory
            )
            vector_store_retriever = connect_vectorstore.as_retriever(search_kwargs={'k': settings.top_retriever_count})

            prompt = PromptTemplate(template=prompt_local, input_variables=["context", "question"])
            memory_stored = ConversationBufferMemory(memory_key="history", input_key="question")
            qa_chain = RetrievalQA.from_chain_type(llm=self.Open_AI,
                                                   chain_type="stuff",
                                                   retriever=vector_store_retriever,
                                                   chain_type_kwargs={"prompt": prompt,
                                                                      "memory": memory_stored},
                                                   return_source_documents=False)

            key_value_pairs = qa_chain.run(ocr_text)

            logger.info(f"Key-value pairs generated successfully.")
            return key_value_pairs

        except Exception as e:
            logger.error("Error generating key-value pairs: %s", str(e), exc_info=True)
            return f"Error generating key-value pairs: {e}"
