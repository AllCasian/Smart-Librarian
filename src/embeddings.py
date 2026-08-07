from openai import OpenAI


class OpenAIEmbeddingService:
    """
    Generates vector embeddings using the OpenAI Embeddings API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_embedding(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate one embedding for one text.

        Args:
            text:
                The text that should be converted into a vector.

        Returns:
            A list of floating-point numbers representing the text.

        Raises:
            ValueError:
                If the text is empty.
        """
        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError(
                "Cannot create an embedding for empty text."
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=normalized_text,
        )

        return response.data[0].embedding

    def create_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in one API request.

        Args:
            texts:
                The texts that should be converted into vectors.

        Returns:
            One embedding for every input text.

        Raises:
            ValueError:
                If the input list is empty or contains empty strings.
        """
        if not texts:
            raise ValueError(
                "The texts list cannot be empty."
            )

        normalized_texts = [
            text.strip()
            for text in texts
        ]

        if any(not text for text in normalized_texts):
            raise ValueError(
                "The texts list cannot contain empty strings."
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=normalized_texts,
        )

        sorted_items = sorted(
            response.data,
            key=lambda item: item.index,
        )

        return [
            item.embedding
            for item in sorted_items
        ]