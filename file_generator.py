import os
from dotenv import load_dotenv
from pathlib import Path
from itertools import islice
from huggingface_hub import login, whoami

from real_esrgan_gen_ai_utils import initialize_model, process_batch


def extract_input_images(input_path: Path) -> set[Path]:
    png_files = set([file.resolve() for file in input_path.glob("*.png")])
    all_pngs = png_files | set([file.resolve() for file in input_path.glob("*.PNG")])
    return all_pngs

def file_generator(input_path: Path, output_path: Path):
    load_dotenv()
    login(token=os.getenv('HF_TOKEN'))
    user_info = whoami()
    print(f" Successfully logged in as: {user_info['name']}")
    png_files = extract_input_images(input_path)
    if not png_files:
        print(f"No png files found in {input_path}")
        return
    else:
        max_batch_size = 10
        truncated_set = set(islice(png_files, max_batch_size))
        print(f"Processing {min(len(png_files),max_batch_size)} png files...")
        model_utils = initialize_model()
        process_batch(model_utils['model'],model_utils['device'], truncated_set, output_path)
