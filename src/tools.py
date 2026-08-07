from src.book_repository import BookRepository


class BookTools:
    """
    Contains local Python functions that can be exposed
    to the language model as tools.
    """

    def __init__(
        self,
        repository: BookRepository,
    ) -> None:
        self.repository = repository

    def get_summary_by_title(
        self,
        title: str,
    ) -> str:
        """
        Return the complete summary for an exact book title.

        Args:
            title:
                The exact title of the book.

        Returns:
            The complete book summary or an informative
            message when the title does not exist.
        """
        normalized_title = title.strip()

        if not normalized_title:
            return "The book title cannot be empty."

        book = self.repository.get_book_by_title(
            normalized_title
        )

        if book is None:
            return (
                "No detailed summary was found for the title "
                f"'{normalized_title}'."
            )

        return book["full_summary"]