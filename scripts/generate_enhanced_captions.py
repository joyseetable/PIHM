# 文件路径: scripts/generate_enhanced_captions_parallel.py

import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import hydra
from omegaconf import DictConfig

# Suppress a specific PIL warning that can occur with very large images
Image.MAX_IMAGE_PIXELS = None

class Blip2CaptionGenerator:
    """A robust class to generate descriptions for images using a local BLIP-2 model."""
    def __init__(self, model_name: str, device: str):
        print(f"Loading BLIP-2 model: {model_name} on device: {device}...")
        self.device = torch.device(device)
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device)
        print(f"BLIP-2 model loaded successfully on {device}.")

    @torch.no_grad()
    def generate_description(self, image: Image.Image, question: str) -> str:
        """Generates a text description for a given image and question."""
        image = image.convert("RGB")
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device, torch.float16)
        
        # Use a controlled generation config
        generated_ids = self.model.generate(**inputs, max_new_tokens=50, num_beams=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

def run_blip2_generation_parallel(cfg: DictConfig):
    """
    Parallel version of the generation script. Each process handles a subset of the data.
    It reads rank and world_size directly from the Hydra config.
    """
    rank = cfg.rank
    world_size = cfg.world_size
    
    # Each process only sees one GPU, which PyTorch will always index as `cuda:0`
    device = "cuda:0"
    
    print(f"----- [Process RANK {rank}/{world_size} on Physical GPU mapped to {device}] Starting... -----")
    generator = Blip2CaptionGenerator(model_name=cfg.caption_generator.model_name, device=device)

    # --- Data Loading and Slicing ---
    annotations_path = hydra.utils.to_absolute_path(cfg.dataset.train_annotations_path)
    with open(annotations_path, 'r') as f:
        original_data = json.load(f)
    
    train_images = [img for img in original_data['images'] if img.get('split') == 'train']
    
    num_images = len(train_images)
    chunk_size = (num_images + world_size - 1) // world_size # Ceiling division
    start_index = rank * chunk_size
    end_index = min(start_index + chunk_size, num_images)
    my_image_list = train_images[start_index:end_index]
    
    print(f"[Process RANK {rank}] Responsible for images {start_index} to {end_index-1}. Total: {len(my_image_list)}.")

    # --- Resume Logic ---
    output_path = annotations_path.replace(".json", f"_part_{rank}.json")
    processed_images = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                processed_images = json.load(f).get("images", [])
        except json.JSONDecodeError:
            print(f"Warning: Partial file {output_path} was corrupted. Starting this part from scratch.")
            processed_images = []
            
    processed_filenames = {img['filename'] for img in processed_images}
    print(f"[Process RANK {rank}] Found {len(processed_filenames)} already processed images in partial file.")

    # --- Main Generation Loop ---
    queries = {
        "global": "Question: What does this image describe from a global perspective? Answer:",
        "background": "Question: What is in the background of this image? Answer:",
        "entity": "Question: What are the main objects or entities present in this image? Answer:"
    }
    
    image_list_to_process = [img for img in my_image_list if img['filename'] not in processed_filenames]

    if not image_list_to_process:
        print(f"[Process RANK {rank}] All assigned images for this part are already processed. Nothing to do.")
        return

    # Initialize tqdm with the correct total for this process
    pbar = tqdm(image_list_to_process, desc=f"Process RANK {rank}", position=rank, total=len(image_list_to_process))
    for image_info in pbar:
        filename = image_info['filename']
        image_path = os.path.join(hydra.utils.to_absolute_path(cfg.dataset.image_dir), filename)
        
        pbar.set_description(f"Processing: {filename}")
        
        if not os.path.exists(image_path):
            # This handles cases where the image file is missing from the folder
            continue
            
        try:
            # Use a 'with' statement for robust file handling
            with Image.open(image_path) as image:
                image.load() # Force loading image data to catch file integrity errors
        except Exception as e:
            print(f"\n[ERROR] RANK {rank}: Could not open or load image {image_path}. Error: {e}. Skipping this image.")
            continue

        enhanced_captions = {}
        for key, query in queries.items():
            try:
                description = generator.generate_description(image, query)
                enhanced_captions[key] = description
            except Exception as e:
                print(f"\n[ERROR] RANK {rank}: Failed to generate caption for image {filename} (query: {key}). Error: {e}. Skipping this caption.")
                enhanced_captions[key] = "[GENERATION FAILED]"
        
        image_info['enhanced_captions'] = enhanced_captions
        processed_images.append(image_info)
        
        # Save progress periodically for safety
        if pbar.n > 0 and pbar.n % 100 == 0:
            with open(output_path, 'w') as f:
                json.dump({"images": processed_images}, f)

    print(f"\n[Process RANK {rank}] Saving final results for this part to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump({"images": processed_images}, f, indent=4)
    print(f"----- [Process RANK {rank}] Generation Finished -----")

@hydra.main(config_path="../configs", config_name="train_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Add rank and world_size to the config if they are not passed from command line
    # This allows the script to be run in single-process mode for debugging
    if 'rank' not in cfg: cfg.rank = 0
    if 'world_size' not in cfg: cfg.world_size = 1
        
    run_blip2_generation_parallel(cfg)

if __name__ == "__main__":
    main()