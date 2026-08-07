from src.conversation import ConversationMemory


def test_memory_stores_messages() -> None:
    memory = ConversationMemory(
        max_messages=4
    )

    memory.add_user_message(
        "Vreau o carte despre magie."
    )

    memory.add_assistant_message(
        "Îți recomand The Hobbit."
    )

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_memory_keeps_only_recent_messages() -> None:
    memory = ConversationMemory(
        max_messages=2
    )

    memory.add_user_message("Message 1")
    memory.add_assistant_message("Message 2")
    memory.add_user_message("Message 3")

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0].content == "Message 2"
    assert messages[1].content == "Message 3"


def test_memory_returns_recent_user_messages() -> None:
    memory = ConversationMemory(
        max_messages=10
    )

    memory.add_user_message("First preference")
    memory.add_assistant_message("First answer")
    memory.add_user_message("Second preference")
    memory.add_assistant_message("Second answer")
    memory.add_user_message("Third preference")

    recent_messages = (
        memory.get_recent_user_messages(
            limit=2
        )
    )

    assert recent_messages == [
        "Second preference",
        "Third preference",
    ]


def test_memory_can_be_cleared() -> None:
    memory = ConversationMemory()

    memory.add_user_message("Hello")
    memory.clear()

    assert memory.get_messages() == []