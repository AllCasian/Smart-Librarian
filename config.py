import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-4.1-mini",
)

IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-1",
)

BOOKS_FILE = PROJECT_ROOT / "data" / "books.json"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "book_summaries"


if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. "
        "Add it to the .env file."
    )
