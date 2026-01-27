from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Envoie un texte et récupère la réponse."""
        pass