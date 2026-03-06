import logging
import time

import pyperclip
import win32api
from pywinauto import Desktop, keyboard

from modules.providers.base_provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

class CopilotUIProvider(LLMProvider):  # pylint: disable=too-few-public-methods
    """LLM provider using browser-based UI automation.

    Interacts with an LLM running inside a browser window (e.g.
    Copilot in Edge) by pasting prompts via the clipboard and
    waiting for the human operator to copy the response back.

    Parameters
    ----------
    window_title : str, optional
        Substring to match against open window titles
        (default ``"Edge"``).

    Raises
    ------
    RuntimeError
        If no window matching *window_title* is found.
    """

    def __init__(self, window_title="Edge"):
        self.window_title = window_title
        self._find_window()
        self.name = "UI LLM translation"

    def _find_window(self):
        """
        Locate and focus the browser window for UI automation.

        Scans all visible desktop windows for one whose title
        contains ``self.window_title``, then brings it to focus.

        Raises
        ------
        RuntimeError
            If no matching window is found.
        """
        windows = Desktop(backend="uia").windows()
        for w in windows:
            if self.window_title in w.window_text():
                self.edge = w
                self.edge.set_focus()
                return
        raise RuntimeError(f"Window containing '{self.window_title}' not found.")

    def _wait_for_click(self):
        """
        Block until the operator performs a physical left-click.

        Polls ``win32api.GetKeyState`` at ~20 Hz until the left
        mouse button state changes, signalling that the operator
        has clicked inside the target input area.
        """
        state_left = win32api.GetKeyState(0x01)
        while True:
            current_state = win32api.GetKeyState(0x01)
            if current_state != state_left:
                return
            time.sleep(0.05)

    def ask(self, content: str, prompt: str) -> str:
        """
        Perform one operator-assisted LLM interaction.

        Copies *prompt* to the clipboard, waits for the operator
        to click inside the browser input, pastes, waits for
        generation, and reads the response from the clipboard.

        Parameters
        ----------
        content : str
            System instructions.  **Not re-sent** each call;
            the operator sets up the system prompt once at the
            start of the browser session.
        prompt : str
            SRT chunk to translate (copied to clipboard).

        Returns
        -------
        str
            The LLM response extracted from the clipboard.
        """
        try:
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
        except (OSError, RuntimeError) as exc:
            raise LLMProviderError(f"UI automation error: {exc}") from exc
