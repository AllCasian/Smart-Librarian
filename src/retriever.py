from src.models import RetrievedBook
from src.vector_store import BookVectorStore


class BookRetriever:
    """
    Retrieves relevant books and formats them as RAG context.
    """

    def __init__(
        self,
        vector_store: BookVectorStore,
        number_of_results: int = 3,
    ) -> None:
        if number_of_results <= 0:
            raise ValueError(
                "number_of_results must be greater than zero."
            )

        self.vector_store = vector_store
        self.number_of_results = number_of_results

    def retrieve(
        self,
        user_query: str,
    ) -> list[RetrievedBook]:
        """
        Retrieve books relevant to the user's request.
        """
        normalized_query = user_query.strip()

        if not normalized_query:
            raise ValueError(
                "User query cannot be empty."
            )

        return self.vector_store.search(
            query=normalized_query,
            number_of_results=self.number_of_results,
        )

    @staticmethod
    def build_context(
        books: list[RetrievedBook],
    ) -> str:
        """
        Format retrieved books as context for the language model.
        """
        if not books:
            return "No relevant books were found."

        context_sections: list[str] = []

        for position, book in enumerate(
            books,
            start=1,
        ):
            section = (
                f"BOOK {position}\n"
                f"Exact title: {book.title}\n"
                f"Author: {book.author}\n"
                f"Book information:\n"
                f"{book.document}"
            )

            context_sections.append(section)

        return "\n\n---\n\n".join(context_sections)