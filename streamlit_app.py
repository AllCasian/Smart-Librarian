import traceback

import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from src.bootstrap import (
    create_image_generation_service,
    create_recommendation_service,
)

from src.language_filter import (
    InappropriateLanguageError,
)

st.set_page_config(
    page_title="Smart Librarian",
    page_icon="📚",
    layout="centered",
)


def initialize_session_state() -> None:
    """
    Initialize all values that must persist between
    Streamlit script reruns.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "recommendation_service" not in st.session_state:
        st.session_state.recommendation_service = (
            create_recommendation_service()
        )

    if "image_generation_service" not in st.session_state:
        st.session_state.image_generation_service = (
            create_image_generation_service()
        )

    if "generate_images" not in st.session_state:
        st.session_state.generate_images = False


def clear_conversation() -> None:
    """
    Clear both the visible Streamlit messages and
    the service's internal conversation memory.
    """
    st.session_state.messages = []

    st.session_state.recommendation_service.clear_conversation()


def display_chat_history() -> None:
    """
    Render text and generated images from the chat history.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            image = message.get("image")

            if image:
                st.image(
                    image,
                    caption=message.get(
                        "image_caption"
                    ),
                    use_container_width=True,
                )


def add_message(
    role: str,
    content: str,
    image: bytes | None = None,
    image_caption: str | None = None,
) -> None:
    """
    Add one message to the visual Streamlit history.
    """

    if not isinstance(content, str):
        raise TypeError(
            "Message content must be a string."
        )

    
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "image": image,
            "image_caption": image_caption,
        }
    )


def get_error_message(
    error: Exception,
) -> str:
    """
    Convert technical exceptions into user-friendly messages.
    """
    if isinstance(error, InappropriateLanguageError):
        return (
            "Mesajul tău conține limbaj inadecvat "
            "și nu poate fi procesat."
        )
    
    if isinstance(error, AuthenticationError):
        return (
            "Cheia OpenAI nu este validă. "
            "Verifică valoarea `OPENAI_API_KEY` din fișierul `.env`."
        )

    if isinstance(error, RateLimitError):
        return (
            "Limita OpenAI API a fost atinsă. "
            "Încearcă din nou mai târziu."
        )

    if isinstance(error, APIConnectionError):
        return (
            "Nu s-a putut realiza conexiunea cu OpenAI API. "
            "Verifică conexiunea la internet."
        )

    if isinstance(error, APIStatusError):
        return (
            "OpenAI API a returnat o eroare "
            f"cu statusul {error.status_code}."
        )

    if isinstance(error, ValueError):
        return str(error)

    if isinstance(error, RuntimeError):
        return (
            "Fluxul de recomandare nu a putut fi finalizat. "
            f"Detalii: {error}"
        )

    return (
        "A apărut o eroare neașteptată. "
        f"Detalii: {error}"
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Despre aplicație")

        st.markdown(
            """
            **Smart Librarian** recomandă o carte folosind:

            1. OpenAI embeddings;
            2. căutare semantică în ChromaDB;
            3. un model GPT;
            4. tool calling pentru rezumat;
            5. generare opțională de imagini.
            """
        )

        st.divider()

        st.session_state.generate_images = st.toggle(
            "Generează o imagine pentru recomandare",
            value=st.session_state.generate_images,
            help=(
                "Poate crește timpul de răspuns și "
                "consumul API."
            ),
        )

        st.divider()

        st.subheader("Exemple")

        st.markdown(
            """
            - Vreau o carte despre magie și prietenie.
            - Vreau să citesc despre libertate și control social.
            - Ce este 1984?
            """
        )

        st.divider()

        if st.button(
            "Șterge conversația",
            use_container_width=True,
        ):
            clear_conversation()
            st.rerun()


def main() -> None:
    st.title("📚 Smart Librarian")

    st.caption(
        "Recomandări de cărți folosind "
        "OpenAI, RAG, ChromaDB și tool calling."
    )

    

    try:
        initialize_session_state()
    except Exception as error:
        st.error(
            "Aplicația nu a putut fi inițializată."
        )

        st.exception(error)
        st.stop()

    render_sidebar()

    display_chat_history()

    user_query = st.chat_input(
        "Descrie ce fel de carte ai vrea să citești..."
    )

    if not user_query:
        return

    normalized_query = user_query.strip()

    if not normalized_query:
        return

    add_message(
        role="user",
        content=normalized_query,
    )

    with st.chat_message("user"):
        st.markdown(normalized_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Caut cele mai potrivite cărți..."
            ):
                recommendation = (
                    st.session_state
                    .recommendation_service
                    .recommend(normalized_query)
                )

        except InappropriateLanguageError as error:
            polite_message = str(error)

            print(
                "Message blocked by the local language filter."
            )

            st.warning(polite_message)

            add_message(
                role="assistant",
                content=polite_message,
            )

            return

        except Exception as error:
            print("Recommendation generation failed.")
            print("Exception type:", type(error))
            print("Exception repr:", repr(error))
            traceback.print_exc()

            error_message = get_error_message(st.error)
            st.error(error_message)

            add_message(
                role="assistant",
                content=error_message,
            )

            return

        st.markdown(recommendation.answer)

        generated_image: bytes | None = None
        image_caption: str | None = None

        if st.session_state.generate_images:
            try:
                with st.spinner(
                    "Generez o ilustrație inspirată de carte..."
                ):
                    generated_image = (
                        st.session_state
                        .image_generation_service
                        .generate_book_image(
                            title=recommendation.title,
                            author=recommendation.author,
                            summary=recommendation.summary,
                        )
                    )

                image_caption = (
                    "Ilustrație originală inspirată de "
                    f"„{recommendation.title}”"
                )

                st.image(
                    generated_image,
                    caption=image_caption,
                    use_container_width=True,
                )

            except Exception:
                print("Image generation failed.")

                st.warning(
                    "Recomandarea a fost generată corect, "
                    "dar imaginea nu a putut fi creată."
                )

    add_message(
        role="assistant",
        content=recommendation.answer,
        image=generated_image,
        image_caption=image_caption,
    )


if __name__ == "__main__":
    main()