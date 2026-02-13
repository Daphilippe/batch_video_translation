import logging
from pathlib import Path
from utils.file_handler import DirectoryMirrorTask
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class SRTOptimizer(DirectoryMirrorTask):
    def __init__(self, input_dir: str, output_dir: str, extensions: tuple = (".srt",)):
        super().__init__(input_dir, output_dir, extensions)

    def process_file(self, input_file: Path) -> None:
        output_file = self.get_output_path(input_file, ".srt")

        with open(input_file, "r", encoding="utf-8") as f:
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