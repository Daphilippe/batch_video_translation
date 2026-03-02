import json
import logging
import random
import re
import time
from pathlib import Path

from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None  # Checked at init time; allows non-legacy engines to load

logger = logging.getLogger(__name__)

class LegacyTranslator(BaseTranslator):
    """Google Translate batch translator with line-level caching.

    Translates SRT files line-by-line through ``deep_translator``,
    applying a technical dictionary for pre-processing and
    maintaining a persistent JSON cache to avoid re-translating
    identical lines across runs.

    Parameters
    ----------
    input_dir : str
        Directory containing source ``.srt`` files.
    output_dir : str
        Directory where translated ``.srt`` files are written.
    config : dict
        Full pipeline config; expects ``config["translation"]``
        with ``source_lang``, ``target_lang``, ``cache_file``,
        and batch tuning parameters.
    """

    def __init__(self, input_dir: str, output_dir: str, config: dict):
        self.config = config
        super().__init__(input_dir, output_dir, extensions=(".srt",))

        if GoogleTranslator is None:
            raise ImportError(
                "deep_translator is required for the legacy (Google) engine. "
                "Install it with: pip install deep-translator"
            )

        self.translator = GoogleTranslator(
            source=self.config["translation"]["source_lang"],
            target=self.config["translation"]["target_lang"]
        )

        # Dictionary preparation
        raw_dict = self.config.get("technical_dictionary", {})
        self.tech_dict = sorted(raw_dict.items(), key=lambda x: len(x[0]), reverse=True)

        # Cache management
        self.cache_path = Path(self.config["translation"]["cache_file"])
        self.cache = self._load_cache()
        self.name = "Legacy translation"

    def _load_cache(self):
        """
        Load the translation cache from disk.

        Reads the JSON file at ``self.cache_path``.  Returns an
        empty dict if the file is missing or malformed.

        Returns
        -------
        dict
            Mapping of SHA-256 line hashes to cached translations.
        """
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                logger.error(f"Failed to load cache: {e}")
        return {}

    def save_cache(self):
        """
        Persist the translation cache to disk.

        Writes ``self.cache`` as indented JSON to ``self.cache_path``,
        creating parent directories if needed.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _apply_dictionary(self, text: str) -> str:
        """
        Replace technical terms using the configured dictionary.

        Matches are case-insensitive and processed longest-first
        to avoid partial replacements.  Each matched term is
        wrapped in parentheses in the output.

        Parameters
        ----------
        text : str
            Source text line to pre-process.

        Returns
        -------
        str
            Text with dictionary terms replaced.
        """
        t = text.lower()
        for key, val in self.tech_dict:
            t = t.replace(key.lower(), f"({val})")
        return t

    def _safe_translate_batch(self, batch: list, _retries: int = 0) -> list:
        """
        Translate a batch of text lines via Google Translate.

        Lines are joined with ``" ||| "`` for a single API call,
        then split back.  Handles HTTP 429 (rate-limit) by
        sleeping and retrying up to ``max_retries`` times.

        Parameters
        ----------
        batch : list of str
            Lines to translate in one request.
        _retries : int, optional
            Internal retry counter (default 0).

        Returns
        -------
        list of str
            Translated lines (same length as *batch*), or an empty
            list on unrecoverable failure.
        """
        max_retries = self.config.get("translation", {}).get("max_retries", 5)
        query = " ||| ".join(batch)
        try:
            result = self.translator.translate(query)
            if not result:
                return []
            return [res.strip() for res in result.split(" ||| ")]
        except Exception as e:  # pylint: disable=broad-exception-caught
            if "429" in str(e):
                if _retries >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for batch. Skipping.")
                    return []
                delay = self.config["translation"].get("retry_delay", 30)
                logger.warning(f"Rate limit hit. Waiting {delay}s... (retry {_retries + 1}/{max_retries})")
                time.sleep(delay)
                return self._safe_translate_batch(batch, _retries + 1)
            logger.error(f"Translation error: {e}")
            return []

    def _separate_cached_lines(self, lines: list[str]) -> tuple[list, list]:
        """Separate structural/cached lines from lines needing translation.

        Analyzes each line: structural lines (indices, timestamps, blanks)
        and already-cached translations are placed directly; remaining
        lines are queued for batch translation.

        Parameters
        ----------
        lines : list of str
            Raw SRT file lines.

        Returns
        -------
        final_lines : list
            Line slots (``str`` for resolved, ``None`` for pending).
        to_translate : list of tuple
            Pending entries as ``(slot_index, pre-treated_text, hash, original_text)``.
        """
        final_lines: list = []
        to_translate: list[tuple] = []

        for line in lines:
            clean = line.strip()
            if re.match(r"^[0-9]+$", clean) or "-->" in clean or not clean:
                final_lines.append(line)
            else:
                l_hash = SRTHandler.get_hash(clean)
                if l_hash in self.cache:
                    final_lines.append(self.cache[l_hash])
                else:
                    pre_treated = self._apply_dictionary(clean)
                    to_translate.append((len(final_lines), pre_treated, l_hash, clean))
                    final_lines.append(None)

        return final_lines, to_translate

    def _retry_untranslated_lines(
        self, current_batch: list, current_meta: list[tuple], results: list
    ) -> None:
        """Retry lines that appear untranslated (identical to source).

        Modifies *results* in-place for lines where the translation
        matches the original text.

        Parameters
        ----------
        current_batch : list of str
            Pre-treated texts sent in this batch.
        current_meta : list of tuple
            Metadata ``(slot_index, hash, original_text)`` per line.
        results : list of str
            Translation results (modified in-place on retry success).
        """
        untranslated = [
            j for j, ((_, _, orig), res) in enumerate(zip(current_meta, results))
            if res.strip().lower() == orig.strip().lower()
        ]
        if not untranslated:
            return
        logger.warning(
            f"Batch: {len(untranslated)}/{len(current_batch)} line(s) "
            f"appear untranslated. Retrying those lines..."
        )
        retry_texts = [current_batch[j] for j in untranslated]
        retry_results = self._safe_translate_batch(retry_texts)
        if retry_results and len(retry_results) == len(retry_texts):
            for k, j in enumerate(untranslated):
                results[j] = retry_results[k]

    def _translate_uncached_lines(self, final_lines: list, to_translate: list[tuple]) -> None:
        """Translate pending lines in batches and fill *final_lines* slots.

        Batches are sized by ``max_chars_batch`` from config.  The
        cache is saved periodically and after the last batch.
        Lines that appear untranslated (identical to source) are
        retried once via ``_retry_untranslated_lines``.

        Parameters
        ----------
        final_lines : list
            Mutable line slots — ``None`` entries are filled in-place.
        to_translate : list of tuple
            Pending entries as ``(slot_index, pre-treated_text, hash, original_text)``.
        """
        i = 0
        batch_count = 0
        max_chars = self.config["translation"].get("max_chars_batch", 2000)

        while i < len(to_translate):
            current_batch, current_meta, current_len = [], [], 0

            while i < len(to_translate) and current_len < max_chars:
                idx, txt, h, orig = to_translate[i]
                current_batch.append(txt)
                current_meta.append((idx, h, orig))
                current_len += len(txt)
                i += 1

            if current_batch:
                results = self._safe_translate_batch(current_batch)
                if results and len(results) == len(current_batch):
                    self._retry_untranslated_lines(current_batch, current_meta, results)
                    for (idx, h, _), res in zip(current_meta, results):
                        final_lines[idx] = res
                        self.cache[h] = res

                batch_count += 1
                if batch_count % 5 == 0:
                    self.save_cache()

                time.sleep(random.uniform(1.2, 2.5))

        if batch_count % 5 != 0:
            self.save_cache()

    def translate_logic(self, text: str):
        """
        Translate full SRT content line-by-line with caching.

        Separates structural lines (indices, timestamps, blanks)
        from text lines.  Cached translations are reused; the
        remainder is sent in batches respecting
        ``max_chars_batch``.

        Parameters
        ----------
        text : str
            Raw SRT file content.

        Returns
        -------
        str
            Translated SRT content.  Lines that failed translation
            are replaced with ``"..."``.
        """
        lines = text.splitlines()
        final_lines, to_translate = self._separate_cached_lines(lines)
        self._translate_uncached_lines(final_lines, to_translate)
        return "\n".join([line if line is not None else "..." for line in final_lines])
