from typing import Any
from unittest import result

import chromadb

from src.embeddings import OpenAIEmbeddingService
from src.models import RetrievedBook


class BookVectorStore:
    """
    Stores and retrieves book documents using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_service: OpenAIEmbeddingService,
    ) -> None:
        self.embedding_service = embedding_service

        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )

        self.collection: Collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": (
                        "Semantic index for book recommendations"
                    )
                },
            )
        )

    @staticmethod
    def build_document(
        book: dict[str, Any],
    ) -> str:
        """
        Convert one book dictionary into the text that will
        be embedded and stored in ChromaDB.
        """
        themes = ", ".join(book["themes"])

        return (
            f"Title: {book['title']}\n"
            f"Author: {book['author']}\n"
            f"Themes: {themes}\n"
            f"Summary: {book['short_summary']}"
        )

    def index_books(
        self,
        books: list[dict[str, Any]],
    ) -> None:
        """
        Generate embeddings and add or update books in ChromaDB.
        """
        if not books:
            raise ValueError(
                "Cannot index an empty list of books."
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for book in books:
            document = self.build_document(book)

            ids.append(book["id"])
            documents.append(document)
            metadatas.append(
                {
                    "title": book["title"],
                    "author": book["author"],
                }
            )

        embeddings = self.embedding_service.create_embeddings(
            documents
        )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        number_of_results: int = 3,
    ) -> list[RetrievedBook]:
        """
        Search for books semantically related to the query.
        """
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if number_of_results <= 0:
            raise ValueError(
            "number_of_results must be greater than zero."
        )

        query_embedding = (
            self.embedding_service.create_embedding(
                normalized_query
            )
        )

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=number_of_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return self._format_search_results(result)

    @staticmethod
    def _format_search_results(
        result: dict[str, Any],
    ) -> list[RetrievedBook]:
        """
        Convert Chroma's nested result into RetrievedBook objects.
        """
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        ids = result.get("ids") or [[]]

        formatted_results: list[RetrievedBook] = []

        for index, document in enumerate(documents[0]):
            metadata = metadatas[0][index] or {}

            title = metadata.get("title")
            author = metadata.get("author")

            if not title or not author:
                continue

            formatted_results.append(
                RetrievedBook(
                    book_id=ids[0][index],
                    title=title,
                    author=author,
                    document=document,
                    distance=distances[0][index],
                )
            )

        return formatted_results

    def count_books(self) -> int:
        """
        Return the number of indexed books.
        """
        return self.collection.count()