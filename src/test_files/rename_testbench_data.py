from pathlib import Path
import os


CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TESTBENCH_DATA_DIR = CURRENT_DIR / "testbench_data"

for file in TESTBENCH_DATA_DIR.glob("*.bin"):
    stem = file.stem
    new_file_stem = stem.split("__")[0]
    new_file = file.with_name(new_file_stem + file.suffix)
    file.rename(new_file)
