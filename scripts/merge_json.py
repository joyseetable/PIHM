# 文件路径: scripts/merge_jsons.py

import json
import os
import glob
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="train_config", version_base="1.2")
def merge_jsons(cfg: DictConfig) -> None:
    """
    Merges the partial JSON files (from parallel generation) into a single
    enhanced annotation file. It combines the generated training data
    with the original validation and test data.
    """
    print("----- Starting to merge partial JSON files -----")
    
    # Use the 'generate' config to find the source files and directories
    annotations_path = hydra.utils.to_absolute_path(cfg.dataset.train_annotations_path)
    base_dir = os.path.dirname(annotations_path)
    base_name = os.path.basename(annotations_path).replace(".json", "")

    # --- Find all partial json files ---
    # This pattern will find _part_0.json, _part_1.json, etc.
    partial_files_pattern = os.path.join(base_dir, f"{base_name}_part_*.json")
    partial_files = sorted(glob.glob(partial_files_pattern))
    
    if not partial_files:
        print(f"Error: No partial files found matching pattern: {partial_files_pattern}")
        print("Please ensure the generation script ran correctly and created the part files.")
        return

    print(f"Found {len(partial_files)} partial files to merge:")
    for f in partial_files:
        print(f" - {f}")

    # --- Collate all enhanced images from the partial files ---
    all_enhanced_images_map = {}
    for file_path in partial_files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                # The file can be a list or a dict with an "images" key
                image_list = data.get("images", []) if isinstance(data, dict) else data
                for item in image_list:
                    # Use filename as the unique key to avoid duplicates
                    all_enhanced_images_map[item['filename']] = item
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file {file_path}. Skipping this file.")

    # --- Combine with original annotations ---
    # We iterate through the original Karpathy JSON to preserve the original order
    # and to add the val/test splits back.
    with open(annotations_path, 'r') as f:
        original_data = json.load(f)
    
    final_images = []
    for image_info in original_data['images']:
        filename = image_info['filename']
        if image_info.get('split') == 'train' and filename in all_enhanced_images_map:
            # If it's a training image and we have enhanced captions for it,
            # add the 'enhanced_captions' field to the original data.
            image_info['enhanced_captions'] = all_enhanced_images_map[filename]['enhanced_captions']
        
        # Add the (potentially modified) image info to our final list
        final_images.append(image_info)
    
    final_data = {"images": final_images}
    
    # --- Save the final merged file ---
    # The final output file name
    output_path = annotations_path.replace(".json", "_enhanced_blip2.json")
    print(f"\nTotal images in merged file: {len(final_images)}")
    print(f"Saving final merged file to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=4)
        
    print("----- Merging complete! -----")
    print(f"The final enhanced annotation file is ready at: {output_path}")
    print("You can now run the 'prepare_flickr30k.py' script to split it.")


if __name__ == "__main__":
    merge_jsons()