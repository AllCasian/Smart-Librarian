import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageFilterResult:
    """
    Represents the result of checking a user message.
    """

    is_allowed: bool
    blocked_terms: tuple[str, ...]


class LanguageFilter:
    """
    Detects locally configured offensive terms.

    The message is checked before it is sent to embeddings,
    ChromaDB or an OpenAI model.
    """

    def __init__(
        self,
        blocked_terms: set[str] | None = None,
    ) -> None:
        default_terms = {
            "idiot",
            "prost",
            "proastă",
            "cretin",
            "cretină",
            "imbecil",
            "imbecilă",
        }

        selected_terms = blocked_terms or default_terms

        self.blocked_terms = {
            term.strip().casefold()
            for term in selected_terms
            if term.strip()
        }

    def check(
        self,
        message: str,
    ) -> LanguageFilterResult:
        """
        Check whether the message contains blocked whole words.
        """
        normalized_message = message.casefold()

        detected_terms = tuple(
            sorted(
                term
                for term in self.blocked_terms
                if self._contains_whole_term(
                    text=normalized_message,
                    term=term,
                )
            )
        )

        return LanguageFilterResult(
            is_allowed=not detected_terms,
            blocked_terms=detected_terms,
        )

    @staticmethod
    def _contains_whole_term(
        text: str,
        term: str,
    ) -> bool:
        """
        Match complete terms instead of arbitrary substrings.
        """
        pattern = (
            r"(?<!\w)"
            + re.escape(term)
            + r"(?!\w)"
        )

        return re.search(
            pattern,
            text,
            flags=re.UNICODE,
        ) is not None

class InappropriateLanguageError(ValueError):
    """
    Raised when a message is blocked by the local language filter.
    """