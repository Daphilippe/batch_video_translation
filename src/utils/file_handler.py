import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DirectoryMirrorTask:
    """Base class for recursive directory-mirroring batch jobs.

    Walks ``input_dir`` for files matching ``extensions``, mirrors the
    relative path structure under ``output_dir``, and delegates each
    file to the subclass's ``process_file`` implementation.

    Parameters
    ----------
    input_dir : str
        Root directory to scan for input files.
    output_dir : str
        Root directory where mirrored output files are written.
    extensions : tuple of str
        Lowercase file extensions to include (e.g. ``(".srt",)``).

    Examples
    --------
    >>> from pathlib import Path
    >>> task = DirectoryMirrorTask("/in", "/out", (".srt",))
    >>> task.input_dir
    PosixPath('/in') if not hasattr(Path, 'drive') else WindowsPath('/in')
    """
    def __init__(self, input_dir: str, output_dir: str, extensions: tuple[str, ...]):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extensions = extensions

    def get_output_path(self, input_file: Path, new_extension: str) -> Path:
        """
        Calculate the mirrored output path for a given input file.

        Preserves the relative directory structure from ``input_dir``
        and replaces the file extension.

        Parameters
        ----------
        input_file : Path
            Absolute path to the source file inside ``input_dir``.
        new_extension : str
            Desired extension for the output file (e.g. ``".srt"``).

        Returns
        -------
        Path
            Target path under ``output_dir`` with the new extension.
        """
        relative_path = input_file.relative_to(self.input_dir)
        target_path = self.output_dir / relative_path
        return target_path.with_suffix(new_extension)

    def run(self) -> None:
        """
        Walk the input directory and process every matching file.

        Iterates recursively over ``input_dir``, filters files by
        ``extensions``, and calls ``process_file`` for each match.
        Logs an error and returns immediately if ``input_dir`` does
        not exist.
        """
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
        """
        Process a single matched file.

        Must be overridden by subclasses to provide the actual
        transformation logic (extraction, transcription, translation,
        etc.).

        Parameters
        ----------
        input_file : Path
            Absolute path to the file to process.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement process_file")
