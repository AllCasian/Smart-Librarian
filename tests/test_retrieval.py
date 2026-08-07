from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from src.embeddings import OpenAIEmbeddingService
from src.vector_store import BookVectorStore
from src.models import RetrievedBook

def print_results(
    query: str,
    results: list[RetrievedBook],
) -> None:
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    for position, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{position}. "
            f"{result.title} "
            f"by {result.author}"
        )

        print(
            f"Distance: "
            f"{result.distance:.4f}"
        )

        print(result.document)


def main() -> None:
    embedding_service = OpenAIEmbeddingService(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
    )

    vector_store = BookVectorStore(
        persist_directory=str(CHROMA_PATH),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_service=embedding_service,
    )

    print(
        f"Indexed books: "
        f"{vector_store.count_books()}"
    )

    test_queries = [
        "Vreau o carte despre magie și prietenie.",
        "Ce recomanzi pentru cineva interesat de război?",
        "Vreau să citesc despre libertate și control social.",
        "Caut o poveste romantică despre prejudecăți.",
        "Vreau o carte despre destin și descoperirea de sine.",
    ]

    for query in test_queries:
        results = vector_store.search(
            query=query,
            number_of_results=3,
        )

        print_results(query, results)


if __name__ == "__main__":
    main()