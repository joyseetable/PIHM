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

class PrototypeGuidedCLIPDataset(Dataset):
   
    def __init__(self, config, split: str, clip_preprocess, clip_tokenize):
        super().__init__()
        self.config = config
        self.split = split
        self.clip_preprocess = clip_preprocess
        self.clip_tokenize = clip_tokenize
        
        
        
        annotations_path = hydra.utils.to_absolute_path(self.config.dataset.get(f"{split}_annotations_path"))
        with open(annotations_path, 'r') as f:
            self.dataset_images = json.load(f)['images']
        self.image_dir_base = self.config.dataset.get("image_dir", "./data/flickr30k/images")
        
        self.ids = []
        for i, d in enumerate(self.dataset_images):
            
            self.ids.extend([(i, j) for j in range(len(d['sentences']))])
        
        
        # if not dist.is_initialized() or dist.get_rank() == 0:
        
        
        self.filenames = [img['filename'] for img in self.dataset_images]
        
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> dict:
        image_idx, caption_idx = self.ids[index]
        image_info = self.dataset_images[image_idx]
        caption_info = image_info['sentences'][caption_idx]
        filename = image_info['filename']
        
       
        image_path = os.path.join(self.image_dir_base, filename)
        
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            
            
            return {
                "image": None,
                "caption": "",
                "filename": filename,
                "image_path": image_path
            }
        
        
        caption = caption_info['raw']
        
       
        image_tensor = self.clip_preprocess(image_pil)
        
      
        return {
            "image": image_tensor,          # [3, 224, 224]
            "caption": caption,             
            "filename": filename,           
            "image_path": image_path,       
            "image_idx": image_idx,         
            "caption_idx": caption_idx      
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
        
        return {
            "images": images,                # [batch_size, 3, 224, 224]
            "text_tokens": text_tokens,      # [batch_size, seq_len]
            "filenames": filenames,         
            "image_paths": image_paths,      
            "captions": captions        
        }

    return collate_fn


def create_data_loader(config, split: str, batch_size: int, 
                      clip_model_name: str = 'ViT-B/32', 
                      num_workers: int = 4,
                      shuffle: bool = None):
    
   
    _, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)
    
   
    tokenizer = clip.tokenize
    
   
    dataset = PrototypeGuidedCLIPDataset(
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