
import os
from pathlib import Path

from file_generator import file_generator

input_path = Path(os.getenv("input_path"))
output_path = Path(os.getenv("output_path"))

file_generator(input_path, output_path)
print(f"Program finished running.")


