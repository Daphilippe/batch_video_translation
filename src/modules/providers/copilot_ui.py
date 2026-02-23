import logging
import time

import pyperclip
import win32api
from pywinauto import Desktop, keyboard

from modules.providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

class CopilotUIProvider(LLMProvider):
    def __init__(self, window_title="Edge"):
        self.window_title = window_title
        self._find_window()
        self.name = "UI LLM translation"

    def _find_window(self):
        windows = Desktop(backend="uia").windows()
        for w in windows:
            if self.window_title in w.window_text():
                self.edge = w
                self.edge.set_focus()
                return
        raise RuntimeError(f"Window containing '{self.window_title}' not found.")

    def _wait_for_click(self):
        """Waits for a physical left click from the user."""
        state_left = win32api.GetKeyState(0x01)
        while True:
            current_state = win32api.GetKeyState(0x01)
            if current_state != state_left:
                return
            time.sleep(0.05)

    def ask(self, content: str, prompt: str) -> str:
        """
        Implementation of the UI-based interaction.

        Note: `content` (system instructions) is intentionally NOT re-sent each call.
        In a browser-based session with a human operator, the context persists across
        messages. The operator sets up the system prompt once at the start of the session.
        Re-sending it every chunk would saturate the LLM's context window.
        """
        # Only copy the SRT chunk to translate — system prompt is already in the conversation
        pyperclip.copy(prompt)

        logger.info("[OPERATOR] Click inside the LLM input box to paste and send...")
        self._wait_for_click()

        # 2. Automation: Paste and Enter
        keyboard.send_keys("^v")
        input("    Press any key to continue...")

        # 3. Wait for manual copy back
        logger.info("[OPERATOR] Wait for generation, then Ctrl+A -> Ctrl+C.")
        input("[OPERATOR] Press ENTER here once you have copied the response...")
        response = pyperclip.paste().strip()
        return response
