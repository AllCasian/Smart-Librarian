from pathlib import Path

from src.book_repository import BookRepository
from src.tools import BookTools


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_FILE = PROJECT_ROOT / "data" / "books.json"


def main() -> None:
    repository = BookRepository(BOOKS_FILE)
    tools = BookTools(repository)

    books = repository.get_all_books()

    print(f"Number of loaded books: {len(books)}")

    summary = tools.get_summary_by_title("1984")

    print("\nSummary returned by the tool:")
    print(summary)

    missing_summary = tools.get_summary_by_title(
        "A Book That Does Not Exist"
    )

    print("\nMissing title response:")
    print(missing_summary)


if __name__ == "__main__":
    main()