import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DirectoryMirrorTask:
    """Base class for recursive directory-mirroring batch jobs.

    Walks ``input_dir`` for files matching ``extensions``, mirrors the
    relative path structure under ``output_dir``, and delegates each file
    to the subclass's ``process_file()`` implementation.
    """
    def __init__(self, input_dir: str, output_dir: str, extensions: tuple[str, ...]):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extensions = extensions

    def get_output_path(self, input_file: Path, new_extension: str) -> Path:
        """Calculates the mirrored output path."""
        relative_path = input_file.relative_to(self.input_dir)
        target_path = self.output_dir / relative_path
        return target_path.with_suffix(new_extension)

    def run(self) -> None:
        """Standardized loop for all batch processing tasks."""
        if not self.input_dir.exists():
            logger.error(f"Source directory not found: {self.input_dir}")
            return

        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(self.extensions):
                    input_file = Path(root) / file
                    # The specific logic will be implemented in subclasses
                    self.process_file(input_file)

    def process_file(self, input_file: Path) -> None:
        """To be overridden by child classes."""
        raise NotImplementedError("Subclasses must implement process_file")
