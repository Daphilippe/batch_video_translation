import logging

import requests

from modules.providers.base_provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class LlamaCPPProvider(LLMProvider):  # pylint: disable=too-few-public-methods
    """LLM provider that communicates with a local llama.cpp server.

    Sends requests to the OpenAI-compatible
    ``/v1/chat/completions`` endpoint exposed by llama.cpp.

    Parameters
    ----------
    url : str, optional
        Base URL of the llama.cpp server
        (default ``"http://127.0.0.1:8080"``).
    """
    def __init__(self, url: str = "http://127.0.0.1:8080"):
        # llama.cpp uses /v1/chat/completions for OpenAI compatibility
        self.url = f"{url}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        self.name = "Local LLM"

    def ask(self, content: str, prompt: str) -> str:
        """
        Send a prompt to the local llama.cpp server.

        Constructs a chat-completion payload with *content* as the
        system message and *prompt* as the user message, then
        posts it to the ``/v1/chat/completions`` endpoint.

        Parameters
        ----------
        content : str
            System-level instructions.
        prompt : str
            User-level prompt (e.g. SRT chunk to translate).

        Returns
        -------
        str
            The assistant's response text.

        Raises
        ------
        LLMProviderError
            On connection errors, empty responses, or unexpected
            response formats.
        """
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": content
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
            result = data['choices'][0]['message']['content']
            if not result or not result.strip():
                raise LLMProviderError("llama.cpp returned an empty response.")
            return result

        except requests.exceptions.RequestException as e:
            raise LLMProviderError(f"Connection error to llama.cpp server: {e}") from e
        except (KeyError, IndexError) as e:
            raise LLMProviderError(f"Invalid response format from llama.cpp: {e}") from e
