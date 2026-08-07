from config import (
    BOOKS_FILE,
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from src.book_repository import BookRepository
from src.embeddings import OpenAIEmbeddingService
from src.vector_store import BookVectorStore


def main() -> None:
    print("Loading books...")

    repository = BookRepository(BOOKS_FILE)
    books = repository.get_all_books()

    print(f"Loaded {len(books)} books.")

    print(
        f"Creating embeddings with model: "
        f"{EMBEDDING_MODEL}"
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

    print("Indexing books in ChromaDB...")

    vector_store.index_books(books)

    print(
        f"Indexing completed. "
        f"Collection contains "
        f"{vector_store.count_books()} books."
    )


if __name__ == "__main__":
    main()