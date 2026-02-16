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

class RSICDDataset(Dataset):
    """
    
    
    {
      "images": [
        {
          "filename": "airport_1.jpg",
          "imgid": 0,
          "sentences": [{"raw": "desc1..."}, {"raw": "desc2..."}],
          "split": "train"  
        },
        ...
      ]
    }
    
    """
    def __init__(self, config, split: str, clip_preprocess, clip_tokenize):
        super().__init__()
        self.config = config
        self.split = split
        self.clip_preprocess = clip_preprocess
        self.clip_tokenize = clip_tokenize
        
        # ---------------------------------------------------------------------
        # 修改点 1: 加载 RSICD 统一的大 JSON 文件
        # ---------------------------------------------------------------------
        # 假设 config 中配置了一个指向 dataset_rsicd.json 的路径
        # 你需要在 hydra 配置里将 key 改为对应的 json 路径
        # 例如: annotations_path: /path/to/dataset_rsicd.json
        annotations_path = hydra.utils.to_absolute_path(
            self.config.dataset.get("annotations_path", "/data/clc/data/RSICD/dataset_rsicd.json")
        )
        
        

        
            
        with open(annotations_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            
        
        images_data = full_data.get('images', full_data) if isinstance(full_data, dict) else full_data

        
        target_split = split 
        
        if split == 'val': 
            target_split = 'val' 
        
        self.annotations = []
        
        for img_item in images_data:
            
            if img_item.get('split') != target_split:
                continue
            
           
            filename = img_item['filename']
            img_id = img_item.get('imgid', 0) 
            
            
            for sent in img_item['sentences']:
                raw_caption = sent['raw']
                
                
                self.annotations.append({
                    "filename": filename,
                    "caption": raw_caption,
                    "imageid": img_id
                })

        
        self.image_dir_base = self.config.dataset.get("image_dir", "/data/clc/data/RSICD/RSICD_images")

        
        
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index: int) -> dict:
        
        ann = self.annotations[index]
        
        filename = ann['filename']
        caption = ann['caption']
        image_id = ann.get('imageid', 0)
        
        
        # image_path = os.path.join(self.image_dir_base, filename)
        base_path = os.path.join(self.image_dir_base, filename)
        
        
        if os.path.exists(base_path):
            image_path = base_path
        else:
            
            
            file_stem = os.path.splitext(filename)[0]
            
            
            potential_paths = [
                os.path.join(self.image_dir_base, file_stem + ".tif"),
                os.path.join(self.image_dir_base, file_stem + ".jpg"),
                os.path.join(self.image_dir_base, file_stem + ".png")
            ]
            
            image_path = base_path 
            for p in potential_paths:
                if os.path.exists(p):
                    image_path = p
                    break
        try:
            
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            
            
            return {
                "image": None,
                "caption": "",
                "filename": filename,
                "image_path": image_path,
                "image_id": image_id,
                "index": index
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
    
    dataset = RSICDDataset(
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