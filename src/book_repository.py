import json
from pathlib import Path
from typing import Any


class BookRepository:
    """
    Repository responsible for reading and querying book data
    stored in a local JSON file.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._books = self._load_books()

    def _load_books(self) -> list[dict[str, Any]]:
        """
        Read and validate the JSON file.

        Returns:
            A list of dictionaries, where every dictionary represents a book.

        Raises:
            FileNotFoundError:
                If the JSON file does not exist.

            ValueError:
                If the JSON content is invalid or does not have
                the expected structure.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Books file was not found: {self.file_path}"
            )

        try:
            with self.file_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                books = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"The books file contains invalid JSON: {error}"
            ) from error

        if not isinstance(books, list):
            raise ValueError(
                "The books JSON file must contain a list of books."
            )

        if len(books) < 10:
            raise ValueError(
                "The assignment requires at least 10 books."
            )

        for index, book in enumerate(books):
            self._validate_book(book, index)

        return books

    @staticmethod
    def _validate_book(
        book: dict[str, Any],
        index: int,
    ) -> None:
        """
        Validate that a book contains all mandatory fields.
        """
        if not isinstance(book, dict):
            raise ValueError(
                f"Book at index {index} must be a JSON object."
            )

        required_fields = {
            "id",
            "title",
            "author",
            "short_summary",
            "full_summary",
            "themes",
        }

        missing_fields = required_fields - book.keys()

        if missing_fields:
            missing_fields_text = ", ".join(sorted(missing_fields))

            raise ValueError(
                f"Book at index {index} is missing fields: "
                f"{missing_fields_text}"
            )

        if not isinstance(book["themes"], list):
            raise ValueError(
                f"The 'themes' field for '{book['title']}' "
                "must be a list."
            )

    def get_all_books(self) -> list[dict[str, Any]]:
        """
        Return all available books.
        """
        return self._books.copy()

    def get_book_by_title(
        self,
        title: str,
    ) -> dict[str, Any] | None:
        """
        Search for a book using a case-insensitive exact title match.

        Example:
            '1984' matches '1984'
            'the hobbit' matches 'The Hobbit'

        Returns:
            The matching book or None if no book is found.
        """
        normalized_title = title.strip().casefold()

        for book in self._books:
            current_title = book["title"].strip().casefold()

            if current_title == normalized_title:
                return book.copy()

        return None