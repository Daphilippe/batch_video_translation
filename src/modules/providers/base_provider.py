from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Raised when an LLM provider fails to produce a valid response.

    Examples
    --------
    >>> raise LLMProviderError("Empty response from server")
    Traceback (most recent call last):
        ...
    modules.providers.base_provider.LLMProviderError: Empty response from server
    """


class LLMProvider(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for LLM backends.

    Every provider must implement ``ask(content, prompt) -> str``.
    Used by ``LLMTranslator`` and ``HybridRefiner`` to decouple
    translation logic from the actual inference backend.
    """

    @abstractmethod
    def ask(self, content: str, prompt: str) -> str:
        """
        Send a prompt to the LLM and return its text response.

        Parameters
        ----------
        content : str
            System-level instructions or context.
        prompt : str
            User-level prompt containing the content to process.

        Returns
        -------
        str
            Raw text response from the LLM.

        Raises
        ------
        LLMProviderError
            If the provider cannot produce a valid response.
        """
