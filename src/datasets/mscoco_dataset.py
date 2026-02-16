import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import hydra
from tqdm import tqdm
import clip
import torch.distributed as dist

class MSCOCODataset(Dataset):
    """
    
    
    [
        {
            "filename": "000000179765.jpg",
            "caption": "A black Honda motorcycle...",
            "imageid": 179765
        },
        ...
    ]
    """
    def __init__(self, config, split: str, clip_preprocess, clip_tokenize):
        super().__init__()
        self.config = config
        self.split = split
        self.clip_preprocess = clip_preprocess
        self.clip_tokenize = clip_tokenize
        
       
        annotations_path = hydra.utils.to_absolute_path(self.config.dataset.get(f"{split}_annotations_path"))
        
        if not os.path.exists(annotations_path):
             raise FileNotFoundError(f"none: {annotations_path}")

        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        
        if split == "train":
           
            self.image_dir_base = self.config.dataset.get("image_dir_base_train", "/data/clc/data/mscoco/train2017")
        else:
            
            self.image_dir_base = self.config.dataset.get("image_dir_base_val", "/data/clc/data/mscoco/val2017")

        
        
        
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index: int) -> dict:
       
        ann = self.annotations[index]
        
        filename = ann['filename']
        caption = ann['caption']
        image_id = ann.get('imageid', 0) 
        
        
        image_path = os.path.join(self.image_dir_base, filename)
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            
            print(f"无法加载图像 {image_path}: {e}")
            return {
                "image": None,
                "caption": "",
                "filename": filename,
                "image_path": image_path
            }
        
        
        image_tensor = self.clip_preprocess(image_pil)
        
        
        return {
            "image": image_tensor,          # [3, 224, 224]
            "caption": caption,             
            "filename": filename,           
            "image_path": image_path,      
            "image_id": image_id,           
            "index": index                  
        }

def create_collate_fn(clip_tokenize):

    def collate_fn(batch):
        
        batch = [b for b in batch if b["image"] is not None and b["caption"] is not None]
        
        if len(batch) == 0:
            return None
        
       
        images = torch.stack([b["image"] for b in batch])
        
       
        captions = [b["caption"] for b in batch]
        text_tokens = clip_tokenize(captions, truncate=True)
        
       
        filenames = [b["filename"] for b in batch]
        image_paths = [b["image_path"] for b in batch]
        
        # -----------------------------------------------------------
        # 
        # -----------------------------------------------------------
        # 
        image_ids = [b["image_id"] for b in batch]
        image_ids_tensor = torch.tensor(image_ids, dtype=torch.long)
        
        return {
            "images": images,                # [batch_size, 3, 224, 224]
            "text_tokens": text_tokens,      # [batch_size, seq_len]
            "filenames": filenames,          # list of strings
            "image_paths": image_paths,      # list of strings
            "captions": captions,            # list of strings
            "image_id": image_ids_tensor     # [batch_size] (Tensor) 
        }
    
    return collate_fn


def create_data_loader(config, split: str, batch_size: int, 
                      clip_model_name: str = 'ViT-B/32', 
                      num_workers: int = 4,
                      shuffle: bool = None):
    
    _, clip_preprocess = clip.load(clip_model_name, device='cpu')
    tokenizer = clip.tokenize
    
    dataset = MSCOCODataset(
        config=config,
        split=split,
        clip_preprocess=clip_preprocess,
        clip_tokenize=tokenizer,
    )
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    collate_fn = create_collate_fn(tokenizer)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader