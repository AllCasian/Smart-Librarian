from src.models import ChatMessage


class ConversationMemory:
    """
    Stores a limited number of user and assistant messages
    for the current application session.
    """

    def __init__(
        self,
        max_messages: int = 10,
    ) -> None:
        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero."
            )

        self.max_messages = max_messages
        self._messages: list[ChatMessage] = []

    def add_user_message(
        self,
        content: str,
    ) -> None:
        """
        Add a user message to the conversation.
        """
        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError(
                "User message cannot be empty."
            )

        self._messages.append(
            ChatMessage(
                role="user",
                content=normalized_content,
            )
        )

        self._trim_history()

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        """
        Add an assistant response to the conversation.
        """
        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError(
                "Assistant message cannot be empty."
            )

        self._messages.append(
            ChatMessage(
                role="assistant",
                content=normalized_content,
            )
        )

        self._trim_history()

    def get_messages(self) -> list[ChatMessage]:
        """
        Return a copy of the current message history.
        """
        return self._messages.copy()

    def get_recent_user_messages(
        self,
        limit: int = 3,
    ) -> list[str]:
        """
        Return the most recent user messages.
        """
        if limit <= 0:
            return []

        user_messages = [
            message.content
            for message in self._messages
            if message.role == "user"
        ]

        return user_messages[-limit:]

    def clear(self) -> None:
        """
        Remove all messages from memory.
        """
        self._messages.clear()

    def _trim_history(self) -> None:
        """
        Keep only the most recent configured messages.
        """
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[
                -self.max_messages:
            ]