import json
from typing import Any

from openai import OpenAI

from src.models import RetrievedBook
from src.retriever import BookRetriever
from src.tool_definitions import (
    GET_SUMMARY_BY_TITLE_TOOL,
)
from src.tools import BookTools
from src.conversation import ConversationMemory
from src.models import (
    BookSummaryResult,
    RecommendationResult
)
from src.language_filter import (
    LanguageFilter,
    InappropriateLanguageError,
)

SYSTEM_PROMPT = """
You are Smart Librarian, a helpful book recommendation assistant.

You receive a user request and a retrieved context containing books
from a local vector database.

You must follow this process:

1. Analyze the user's request.
2. Select exactly one book from the retrieved context.
3. Never select a book that is not present in the context.
4. Copy the exact title from the context.
5. Call get_summary_by_title with that exact title.
6. After receiving the tool result, provide the final answer.

The final answer must:
- be written in the same language as the user;
- clearly mention the exact book title and author;
- briefly explain why the book matches the request;
- include the detailed summary returned by the tool;
- distinguish the recommendation reason from the detailed summary;
- never invent information outside the retrieved context or tool result.
""".strip()


class BookRecommendationService:
    """
    Coordinates RAG retrieval, OpenAI tool calling,
    local tool execution and final answer generation.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        retriever: BookRetriever,
        book_tools: BookTools,
        memory: ConversationMemory,
        language_filter: LanguageFilter,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retriever = retriever
        self.book_tools = book_tools
        self.memory = memory
        self.language_filter = language_filter

    def recommend(
        self,
        user_query: str,
    ) -> RecommendationResult:
        """
        Generate a contextual recommendation and detailed summary.

        Flow:
        1. Build a retrieval query using recent conversation context.
        2. Retrieve relevant books.
        3. Ask the model to select one and call the tool.
        4. Validate and execute the local tool.
        5. Ask the model for the final answer.
        6. Store the conversation turn in memory.
        """
        normalized_query = user_query.strip()

        if not normalized_query:
            raise ValueError(
                "User query cannot be empty."
            )

        filter_result = self.language_filter.check(
            normalized_query
        )

        if not filter_result.is_allowed:
            raise InappropriateLanguageError(
                "Te rog să reformulezi mesajul într-un limbaj politicos. "
                "Pot continua imediat ce întrebarea nu mai conține "
                "formulări ofensatoare."
        )

        retrieval_query = self._build_retrieval_query(
            current_query=normalized_query
        )

        retrieved_books = self.retriever.retrieve(
            retrieval_query
        )

        if not retrieved_books:
            raise RuntimeError(
                "No books were retrieved from ChromaDB."
            )

        rag_context = self.retriever.build_context(
            retrieved_books
        )

        messages = self._build_model_messages(
            user_query=normalized_query,
            rag_context=rag_context,
        )

        tool_response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[
                    GET_SUMMARY_BY_TITLE_TOOL,
                ],
                tool_choice="required",
                temperature=0.2,
            )
        )

        assistant_message = (
            tool_response.choices[0].message
        )

        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            raise RuntimeError(
                "The model did not request a tool call."
            )

        if len(tool_calls) != 1:
            raise RuntimeError(
                "The model must call exactly one tool."
            )

        tool_call = tool_calls[0]

        summary_result = self._execute_tool_call(
            tool_call=tool_call,
            retrieved_books=retrieved_books,
        )

        messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content":  summary_result.serialized_content,
            }
        )

        final_response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )
        )

        final_content = (
            final_response.choices[0].message.content
        )

        if not final_content:
            raise RuntimeError(
                "The model returned an empty final response."
            )

        final_answer = final_content.strip()

        self.memory.add_user_message(
            normalized_query
        )

        self.memory.add_assistant_message(
            final_answer
        )

        return RecommendationResult(
            title=summary_result.title,
            author=summary_result.author,
            summary=summary_result.summary,
            answer=final_answer,
        )

    @staticmethod
    def _build_user_message(
        user_query: str,
        context: str,
    ) -> str:
        """
        Create the message containing the original request
        and the RAG results.
        """
        return (
            "CURRENT USER REQUEST:\n"
            f"{user_query}\n\n"
            "RETRIEVED BOOK CONTEXT:\n"
            f"{context}\n\n"
            "Select the single best book and call "
            "get_summary_by_title with its exact title."
        )

    def _execute_tool_call(
        self,
        tool_call: Any,
        retrieved_books: list[RetrievedBook],
    ) -> BookSummaryResult:
        """
        Validate and execute the tool requested by the model.
        """
        function_name = tool_call.function.name

        if function_name != "get_summary_by_title":
            raise RuntimeError(
                f"Unsupported tool requested: {function_name}"
            )

        arguments = self._parse_tool_arguments(
            tool_call.function.arguments
        )

        title = arguments.get("title")

        if not isinstance(title, str):
            raise RuntimeError(
                "Tool argument 'title' must be a string."
            )

        normalized_title = title.strip()

        if not normalized_title:
            raise RuntimeError(
                "Tool argument 'title' cannot be empty."
            )

        selected_book = self._find_retrieved_book(
            title=normalized_title,
            retrieved_books=retrieved_books,
        )

        summary = self.book_tools.get_summary_by_title(
            selected_book.title
        )

        serialized_content = json.dumps(
            {
                "title": selected_book.title,
                "author": selected_book.author,
                "summary": summary,
            },
            ensure_ascii=False,
        )

        return BookSummaryResult(
            title=selected_book.title,
            author=selected_book.author,
            summary=summary,
            serialized_content=serialized_content,
        )

    @staticmethod
    def _parse_tool_arguments(
        raw_arguments: str,
    ) -> dict[str, Any]:
        """
        Convert the JSON arguments generated by the model
        into a Python dictionary.
        """
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "The model returned invalid tool arguments."
            ) from error

        if not isinstance(arguments, dict):
            raise RuntimeError(
                "Tool arguments must be a JSON object."
            )

        return arguments

    @staticmethod
    def _find_retrieved_book(
        title: str,
        retrieved_books: list[RetrievedBook],
    ) -> RetrievedBook:
        """
        Ensure that the selected book came from the RAG results.
        """
        normalized_title = title.casefold()

        for book in retrieved_books:
            if book.title.casefold() == normalized_title:
                return book

        allowed_titles = ", ".join(
            book.title
            for book in retrieved_books
        )

        raise RuntimeError(
            "The model selected a title outside the "
            "retrieved RAG context. "
            f"Allowed titles: {allowed_titles}"
        )

    def _build_retrieval_query(
        self,
        current_query: str,
    ) -> str:
        """
        Combine the current request with a small amount
        of recent user context for semantic retrieval.
        """
        previous_user_messages = (
            self.memory.get_recent_user_messages(
                limit=2
            )
        )

        if not previous_user_messages:
            return current_query

        previous_context = "\n".join(
            f"- {message}"
            for message in previous_user_messages
        )

        return (
            "Previous user preferences:\n"
            f"{previous_context}\n\n"
            "Current user request:\n"
            f"{current_query}"
        )

    def _build_model_messages(
        self,
        user_query: str,
        rag_context: str,
    ) -> list[dict[str, Any]]:
        """
        Construct system, conversation and current RAG messages.
        """
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        for message in self.memory.get_messages():
            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": self._build_user_message(
                    user_query=user_query,
                    context=rag_context,
                ),
            }
        )

        return messages

    def clear_conversation(self) -> None:
        """
        Clear the current in-memory chat history.
        """
        self.memory.clear()