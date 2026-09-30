import gc
import os
import time
from pathlib import Path

#  --- Monkey patch for HUGGINGFACE COMPATIBILITY ---
import huggingface_hub
from huggingface_hub import hf_hub_download

if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
#  ---------------------------------------------------------------------------

import torch
from PIL import Image
from py_real_esrgan.model import RealESRGAN

def initialize_model() -> dict:
    """Initializes device, allocates hardware, and pre-loads models weights."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)

    # model.load_weights('weights/RealESRGAN_x4.pth')
    file_path = hf_hub_download(
        repo_id="sberbank-ai/Real-ESRGAN",
        filename="RealESRGAN_x4.pth",
        repo_type="model"
    )
    print(f"Using device: {device}")

    model = RealESRGAN(device, scale=4)
    model.load_weights(file_path)



    return {"model": model, "device": device}

def process_batch(model, device, images_set: set[Path],output_dir):
    """Processes up to 10 images sequentially with memory cleanup and saves them to output_dir."""
    image_list = list(images_set)
    os.makedirs(output_dir, exist_ok=True)
    # Turn off gradient calculation globally
    with torch.no_grad():
        for path in image_list:
            if not os.path.exists(path):
                print(f"Skipping: {path} not found.")
            else:
                print(f"Processing... {os.path.basename(path)}")
                # Context-managed image loading to avoid lingering handles
                start = time.time()
                with Image.open(path) as img:
                    print(f"Image opened in {time.time() - start:.2f}s")
                    rgb_img = img.convert('RGB')
                    print("Running ESRGAN prediction...")
                    sr_image = model.predict(rgb_img)
                # Save the output
                img_output_name = f"enhanced_{os.path.basename(path)}"
                print(f"the path of the image should be at: {img_output_name}")
                sr_image.save(os.path.join(output_dir, img_output_name))
                # Explicit memory cleanup for hardware efficiency
                del sr_image
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
