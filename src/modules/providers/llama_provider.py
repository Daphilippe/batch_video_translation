import requests
import logging
from modules.providers.base_provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

class LlamaCPPProvider(LLMProvider):
    def __init__(self, url="http://127.0.0.1:8080"):
        # llama.cpp utilise /v1/chat/completions pour la compatibilité OpenAI
        self.url = f"{url}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        self.name = "Local LLM"

    def ask(self, content:str, prompt: str) -> str:
        """
        Méthode requise par ton LLMTranslator.
        Envoie le prompt au serveur local llama.cpp.
        """
        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": f"{content}"
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False,
            "repeat_penalty": 1.1
        }
        try:
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=180)
            response.raise_for_status()
            data = response.json()
            # Extraction du contenu du message
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(f"Connection error to llama.cpp server: {e}") from e
        except (KeyError, IndexError) as e:
            raise LLMProviderError(f"Invalid response format from llama.cpp: {e}") from e