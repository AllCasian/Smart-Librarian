from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from src.bootstrap import create_recommendation_service

def print_welcome_message() -> None:
    print("=" * 70)
    print("Smart Librarian")
    print("=" * 70)
    print(
        "Describe the kind of book you would like to read."
    )
    print()
    print("Examples:")
    print(
        "- Vreau o carte despre libertate și control social."
    )
    print(
        "- Ce recomanzi dacă iubesc poveștile fantastice?"
    )
    print(
        "- Caut o poveste despre război și prietenie."
    )
    print(
        "- Ce este 1984?"
    )
    print()
    print("Commands:")
    print("- clear: clear the conversation history")
    print("- exit: close the application")
    print("=" * 70)


def main() -> None:
    try:
        recommendation_service = (
            create_recommendation_service()
        )
    except Exception as error:
        print(
            "The application could not be initialized."
        )
        print(f"Technical details: {error}")
        return

    print_welcome_message()

    while True:
        user_query = input("\nYou: ").strip()

        normalized_command = user_query.casefold()

        if normalized_command in {
            "exit",
            "quit",
        }:
            print("\nSmart Librarian: Goodbye!")
            break

        if normalized_command in {
            "clear",
            "reset",
        }:
            recommendation_service.clear_conversation()

            print(
                "\nSmart Librarian: "
                "Conversation history cleared."
            )
            continue

        if not user_query:
            print(
                "\nSmart Librarian: "
                "Please enter a book preference."
            )
            continue

        try:
            final_answer = (
                recommendation_service.recommend(
                    user_query
                )
            )
        except ValueError as error:
            print(
                "\nSmart Librarian: "
                f"{error}"
            )
            continue
        except Exception as error:
            print(
                "\nSmart Librarian: "
                "I could not generate a recommendation."
            )
            print(f"Technical details: {error}")
            continue

        recommendation = (
            recommendation_service.recommend(
                user_query
            )
        )

        print("\nSmart Librarian:")
        print(recommendation.answer)


if __name__ == "__main__":
    main()