import base64

from openai import OpenAI


class ImageGenerationService:
    """
    Generates an original representative image for a book.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-image-1",
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_book_image(
        self,
        title: str,
        author: str,
        summary: str,
    ) -> bytes:
        """
        Generate an original illustration based on a book summary.

        The result is returned as raw image bytes so it can
        be displayed directly by Streamlit.
        """
        normalized_title = title.strip()
        normalized_author = author.strip()
        normalized_summary = summary.strip()

        if not normalized_title:
            raise ValueError(
                "Book title cannot be empty."
            )

        if not normalized_summary:
            raise ValueError(
                "Book summary cannot be empty."
            )

        prompt = self._build_prompt(
            title=normalized_title,
            author=normalized_author,
            summary=normalized_summary,
        )

        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size="1024x1024",
            quality="medium",
            n=1,
        )

        if not response.data:
            raise RuntimeError(
                "The image API returned no image data."
            )

        encoded_image = response.data[0].b64_json

        if not encoded_image:
            raise RuntimeError(
                "The image API returned an empty image."
            )

        return base64.b64decode(encoded_image)

    @staticmethod
    def _build_prompt(
        title: str,
        author: str,
        summary: str,
    ) -> str:
        """
        Build a safe prompt for an original editorial illustration.
        """
        return (
            "Create an original, atmospheric editorial illustration "
            "inspired by the following book. The image should represent "
            "a central theme, location or emotional atmosphere from the "
            "summary.\n\n"
            f"Book title: {title}\n"
            f"Author: {author}\n"
            f"Summary: {summary}\n\n"
            "Requirements:\n"
            "- Create a new, original composition.\n"
            "- Do not reproduce or imitate an existing book cover.\n"
            "- Do not include the book title, author name, logos, "
            "publisher marks or any other text.\n"
            "- Avoid recognizable actor likenesses.\n"
            "- Use a polished cinematic storybook aesthetic.\n"
            "- Focus on atmosphere and symbolism rather than an exact "
            "literal recreation.\n"
            "- Square composition suitable for display in a web app."
        )