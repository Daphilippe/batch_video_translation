"""Shared test helpers — single MockProvider definition for all test modules."""

from modules.providers.base_provider import LLMProvider


class MockProvider(LLMProvider):
    """Test double that returns pre-configured responses in order.

    Parameters
    ----------
    responses : list, optional
        Ordered list of return values.  If an element is an
        ``Exception``, it is raised instead of returned.
    """

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.call_count = 0
        self.prompts: list[str] = []
        self.name = "MockLLM"

    def ask(self, content: str, prompt: str) -> str:
        """Return the next pre-configured response or empty string."""
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        self.call_count += 1
        return ""
