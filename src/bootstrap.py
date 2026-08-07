from config import (
    BOOKS_FILE,
    CHAT_MODEL,
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    IMAGE_MODEL,
    OPENAI_API_KEY,
)
from src.book_repository import BookRepository
from src.conversation import ConversationMemory
from src.embeddings import OpenAIEmbeddingService
from src.recommendation_service import (
    BookRecommendationService,
)
from src.retriever import BookRetriever
from src.tools import BookTools
from src.vector_store import BookVectorStore
from src.image_generation import (
    ImageGenerationService,
)
from src.language_filter import LanguageFilter

language_filter = LanguageFilter()

def create_recommendation_service(
) -> BookRecommendationService:
    """
    Construct and connect all application dependencies.

    The same factory is used by both the CLI interface
    and the Streamlit interface.
    """
    repository = BookRepository(
        BOOKS_FILE
    )

    book_tools = BookTools(
        repository=repository
    )

    embedding_service = OpenAIEmbeddingService(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
    )

    vector_store = BookVectorStore(
        persist_directory=str(CHROMA_PATH),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_service=embedding_service,
    )

    if vector_store.count_books() == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty. "
            "Run: python -m scripts.index_books"
        )

    retriever = BookRetriever(
        vector_store=vector_store,
        number_of_results=3,
    )

    memory = ConversationMemory(
        max_messages=10
    )

    return BookRecommendationService(
        api_key=OPENAI_API_KEY,
        model=CHAT_MODEL,
        retriever=retriever,
        book_tools=book_tools,
        memory=memory,
        language_filter=language_filter,
    )

def create_image_generation_service(
) -> ImageGenerationService:
    """
    Create the service used for book illustrations.
    """
    return ImageGenerationService(
        api_key=OPENAI_API_KEY,
        model=IMAGE_MODEL,
    )