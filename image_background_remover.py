from pathlib import Path
from typing import Set

from rembg import remove
from PIL import Image

def remove_image_backgrounds(input_paths: Set[Path], output_dir: Path):
    for path in input_paths:
        try:
            input_image = Image.open(path)
            output_image = remove(input_image)
            output_image.save(output_dir)
        except FileNotFoundError as e:
            print(f"Error removing image background: {e}")