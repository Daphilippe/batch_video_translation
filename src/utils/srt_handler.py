import re
import hashlib
import unicodedata
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SRTHandler:
    TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")

    @staticmethod
    def clean_text(text: str) -> str:
        replacements = {"**": "", "": "", "□": "-", "▪": "-", "…": "..."}
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return " ".join(text.split()).strip()

    @staticmethod
    def canonicalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("\ufeff") # Remove BOM
        for ch in ["\u200b", "\u200c", "\u200d", "\xa0"]:
            text = text.replace(ch, "")
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def get_hash(cls, text: str) -> str:
        return hashlib.sha256(cls.canonicalize(text).encode("utf-8")).hexdigest()

    @classmethod
    def extract_timestamps(cls, path: Path) -> list:
        if not path.exists(): return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if "-->" in line and cls.TIMESTAMP_RE.fullmatch(line.strip())]

    @classmethod
    def normalize_file(cls, path_in: Path, path_out: Path):
        """Rebuilds the SRT to ensure perfect 1, 2, 3 numbering and clean spacing."""
        with open(path_in, "r", encoding="utf-8") as f:
            content = f.read()
        
        blocks = []
        current_ts = None
        current_text = []
        
        for line in content.splitlines():
            line = line.strip()
            if not line: continue
            if "-->" in line and cls.TIMESTAMP_RE.fullmatch(line):
                if current_ts and current_text:
                    blocks.append((current_ts, " ".join(current_text)))
                current_ts, current_text = line, []
            elif not line.isdigit():
                current_text.append(line)
        
        if current_ts and current_text:
            blocks.append((current_ts, " ".join(current_text)))

        output = [f"{i}\n{ts}\n{txt}\n" for i, (ts, txt) in enumerate(blocks, 1)]
        final_text = "\n".join(output)

        with open(path_out, "w", encoding="utf-8") as f:
            f.write(final_text)
            
    @staticmethod
    def parse_to_blocks(content: str) -> list:
        """Parses SRT content with aggressive cleaning of LLM artifacts."""
        blocks = []
        current = {"index": None, "start": None, "end": None, "text": []}

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

            # 1. Match Index
            if re.match(r"^\d+$", line):
                if current["index"] is not None:
                    blocks.append(current)
                current = {"index": int(line), "start": None, "end": None, "text": []}

            # 2. Match Timestamps
            elif "-->" in line:
                times = [x.strip() for x in line.split("-->")]
                if len(times) == 2:
                    current["start"], current["end"] = times[0], times[1]

            # 3. Match Text (The problematic part)
            else:
                # CLEANING: Remove typical LLM artifacts like ["text"] or ['text']
                # This regex removes leading/trailing brackets and quotes
                clean_line = re.sub(r"^['\"\[\s]+|['\"\]\s]+$", "", line)
                
                if clean_line:
                    current["text"].append(clean_line)

        if current["index"] is not None:
            blocks.append(current)
            
        return blocks

    @staticmethod
    def merge_identical_blocks(blocks: list) -> list:
        """Merges consecutive blocks if the text is identical."""
        merged = []
        prev = None

        for b in blocks:
            text = "\n".join(b["text"]).strip()
            if prev and text == prev["text"]:
                prev["end"] = b["end"]
            else:
                prev = {"start": b["start"], "end": b["end"], "text": text}
                merged.append(prev)
        return merged

    @staticmethod
    def render_blocks(blocks: list) -> str:
        """Converts blocks back into a valid SRT string, ensuring text is a string."""
        out = []
        for i, b in enumerate(blocks, start=1):
            # Ensure text is joined properly and NOT a string representation of a list
            if isinstance(b['text'], list):
                text_content = "\n".join(b['text']).strip()
            else:
                text_content = str(b['text']).strip()

            out.append(f"{i}")
            out.append(f"{b['start']} --> {b['end']}")
            out.append(text_content)
            out.append("") # Empty line between blocks
            
        return "\n".join(out)