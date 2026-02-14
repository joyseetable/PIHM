# 文件路径: scripts/prepare_flickr30k.py

import json
from tqdm import tqdm
import os

def split_enhanced_karpathy_annotations(
    enhanced_json_path: str, 
    output_dir: str
):
    """
    Splits the large, enhanced Karpathy-style annotation JSON into separate
    train, val, and test JSON files. It ensures the enhanced captions
    are correctly placed within the new train.json.
    """
    if not os.path.exists(enhanced_json_path):
        print(f"Error: Enhanced JSON file not found at '{enhanced_json_path}'")
        print("Please ensure that you have run the merging script ('merge_jsons.py') first.")
        return

    print(f"Loading ENHANCED Karpathy annotations from: {enhanced_json_path}")
    with open(enhanced_json_path, 'r') as f:
        data = json.load(f)

    # Create dictionaries to hold the structure for each split's JSON file
    split_data = {
        "train": {"images": []},
        "val": {"images": []},
        "test": {"images": []}
    }

    print("Splitting images into train, val, and test sets...")
    image_count = 0
    for image_info in tqdm(data['images'], desc="Processing images"):
        # The 'split' key is the ground truth from the original Karpathy file
        split = image_info.get('split')
        
        if split in split_data:
            # Append the entire image_info dictionary to the correct list
            split_data[split]['images'].append(image_info)
            image_count += 1
        elif split == 'restval':
            # As per Karpathy's convention, 'restval' images are part of the training set
            split_data['train']['images'].append(image_info)
            image_count += 1
        else:
            print(f"Warning: Found an unknown split type '{split}' for image {image_info.get('filename')}. Skipping.")

    print(f"Successfully processed {image_count} images.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split to its own new JSON file
    for split, content in split_data.items():
        output_path = os.path.join(output_dir, f"{split}.json")
        num_images = len(content['images'])
        
        # A final check, especially for the test set
        if split == 'test' and num_images == 0:
            print(f"\nWarning: The generated {split}.json is empty.")
            print("This might happen if the source JSON has an unusual structure.")
            print("Please double-check your source file.")
        
        print(f"Saving '{split}' split with {num_images} images to: {output_path}")
        with open(output_path, 'w') as f:
            # Use indent=4 for readability
            json.dump(content, f, indent=4)
            
    print("\n----- Flickr30k annotation splitting finished! -----")
    print("Your final `train.json`, `val.json`, and `test.json` files are ready.")


if __name__ == '__main__':
    # --- Configuration ---
    # This script reads the ENHANCED file...
    # IMPORTANT: Ensure this path points to the file created by `merge_jsons.py`
    enhanced_json_path = "/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/annotations/dataset_flickr30k_enhanced_blip2.json"
    
    # ...and saves the split files to the same directory.
    output_dir = "/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/annotations/"
    
    # --- Run ---
    split_enhanced_karpathy_annotations(enhanced_json_path, output_dir)