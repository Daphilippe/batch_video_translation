from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Raised when an LLM provider fails to produce a valid response."""
    pass


class LLMProvider(ABC):
    @abstractmethod
    def ask(self, content: str, prompt: str) -> str:
        """Envoie un texte et récupère la réponse."""
        pass