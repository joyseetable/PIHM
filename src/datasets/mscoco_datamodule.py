# src/datamodules/global_feature_datamodule.py

import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import hydra
import clip
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Subset
import numpy as np

# -----------------------------------------------------------

# -----------------------------------------------------------
from .mscoco_dataset import MSCOCODataset, create_collate_fn

class MSCOCODatamodule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_transform = None
        self.tokenizer = clip.tokenize
        self._collate_fn = None
        
        if dist.is_initialized() and dist.get_rank() == 0:
            print("COCO Flat Version")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        
        
        if self.image_transform is None:
            clip_model_name = self.config.model.get('clip_model_name', 'ViT-B/32')
            
            _, self.image_transform = clip.load(
                clip_model_name, 
                device="cpu", 
                jit=False
            )
        
        
        if self._collate_fn is None:
            self._collate_fn = create_collate_fn(self.tokenizer)
        
        
        
        
        if stage == 'fit' or stage is None:
            full_train_dataset = MSCOCODataset(
                config=self.config, 
                split='train',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            train_ratio = self.config.dataset.get('train_ratio', 1.0) 
            
            if train_ratio < 1.0:
                total_size = len(full_train_dataset)
                subset_size = int(total_size * train_ratio)
                rng = np.random.RandomState(42) 
                indices = rng.permutation(total_size)
                selected_indices = indices[:subset_size]
                
                
                self.train_dataset = Subset(full_train_dataset, selected_indices)
                
                
                # if not dist.is_initialized() or dist.get_rank() == 0:
                    
            else:
                self.train_dataset = full_train_dataset
                if not dist.is_initialized() or dist.get_rank() == 0:
                     print(f"✅ [Data Scaling]: {len(self.train_dataset)}")
            self.val_dataset = MSCOCODataset(
                config=self.config, 
                split='val',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"train: {len(self.train_dataset)}, val: {len(self.val_dataset)}")
            
        if stage == 'test' or stage is None:
           
            test_split = 'test' if 'test_annotations_path' in self.config.dataset else 'val'
            
            self.test_dataset = MSCOCODataset(
                config=self.config, 
                split=test_split,
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"test: {len(self.test_dataset)} (Split: {test_split})")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.dataset.batch_size, 
            
            shuffle=(sampler is None),
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            drop_last=True  
        )

    def val_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            
            sampler = DistributedSampler(
                self.val_dataset, 
                num_replicas=1, 
                rank=0,         
                shuffle=False
            )
            
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.dataset.batch_size, 
            shuffle=False,
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            # -----------------------------------------------------------
            # 
            # -----------------------------------------------------------
            drop_last=False 
        )

    def test_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.test_dataset, 
                num_replicas=1, 
                rank=0,         
                shuffle=False   
            )
            
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config.dataset.batch_size,
            shuffle=False, 
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            # -----------------------------------------------------------
            # 
            # -----------------------------------------------------------
            drop_last=False
        )