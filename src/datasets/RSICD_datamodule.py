# src/datamodules/global_feature_datamodule.py

import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import hydra
import clip
import torch.distributed as dist
from torch.utils.data import DistributedSampler


from .RSICD_dataset import RSICDDataset, create_collate_fn

class RSICDDatamodule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_transform = None
        self.tokenizer = clip.tokenize
        self._collate_fn = None
        
        # if dist.is_initialized() and dist.get_rank() == 0:
           

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
            self.train_dataset = RSICDDataset(
                config=self.config, 
                split='train',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            
            self.val_dataset = RSICDDataset(
                config=self.config, 
                split='val',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[Setup] : {len(self.train_dataset)},: {len(self.val_dataset)}")
            
        if stage == 'test' or stage is None:

            self.test_dataset = RSICDDataset(
                config=self.config, 
                split='test',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )


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
            drop_last=False
        )