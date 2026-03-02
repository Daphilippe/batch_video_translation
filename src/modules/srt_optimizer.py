import logging
from pathlib import Path

from utils.file_handler import DirectoryMirrorTask
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class SRTOptimizer(DirectoryMirrorTask):
    """Step 3 of the pipeline — SRT structural standardization.

    Applies ``SRTHandler.standardize`` to each raw transcription file:
    merge identical consecutive blocks, clean LLM artifacts, remove
    empty blocks, and re-index.

    Parameters
    ----------
    input_dir : str
        Directory containing raw ``.srt`` files.
    output_dir : str
        Directory where standardized ``.srt`` files are written.
    extensions : tuple of str, optional
        File extensions to process (default ``(".srt",)``).
    """

    def __init__(self, input_dir: str, output_dir: str, extensions: tuple = (".srt",)):
        super().__init__(input_dir, output_dir, extensions)

    def process_file(self, input_file: Path) -> None:
        """
        Standardize a single SRT file.

        Reads *input_file*, applies the full ``SRTHandler.standardize``
        pipeline, and writes the result to the mirrored output path.

        Parameters
        ----------
        input_file : Path
            Path to the raw ``.srt`` file.
        """
        output_file = self.get_output_path(input_file, ".srt")

        with open(input_file, encoding="utf-8") as f:
            content = f.read()

        # Apply the full conditioning immediately
        standardized_content = SRTHandler.standardize(content)

        logger.info(f"[STEP 3] Standardizing structure: {input_file.name}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(standardized_content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SRT Redundancy Optimizer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    optimizer = SRTOptimizer(args.input, args.output)
    optimizer.run()
