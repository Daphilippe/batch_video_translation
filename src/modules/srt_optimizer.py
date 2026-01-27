import logging
from pathlib import Path
from utils.file_handler import DirectoryMirrorTask
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class SRTOptimizer(DirectoryMirrorTask):
    def __init__(self, input_dir, output_dir, extensions=(".srt",)):
        super().__init__(input_dir, output_dir, extensions)

    def process_file(self, input_file: Path):
        output_file = self.get_output_path(input_file, ".srt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f:
            original_content = f.read()

        # 1. Parse & Merge logic
        blocks = SRTHandler.parse_to_blocks(original_content)
        merged_blocks = SRTHandler.merge_identical_blocks(blocks)
        optimized_content = SRTHandler.render_blocks(merged_blocks)

        # 2. Check if anything actually changed
        if SRTHandler.get_hash(original_content) != SRTHandler.get_hash(optimized_content):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(optimized_content)
            logger.info(f"[OPTIMIZED] {input_file.name}")
        else:
            logger.info(f"[NO CHANGE] {input_file.name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SRT Redundancy Optimizer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    optimizer = SRTOptimizer(args.input, args.output)
    optimizer.run()