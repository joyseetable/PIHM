import os
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, Callback 
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
from typing import List
from copy import deepcopy 


from src.datasets import create_datamodule 
from systems.system import DistillSystem

from src.utils.logging_callbacks import TextLoggingCallback
# from src.callbacks.schdule import DynamicLossWeightScheduler
from omegaconf import OmegaConf, DictConfig

# ==========================================

# ==========================================
class EMACallback(Callback):

    def __init__(self, decay=0.999, validate_original_weights=False):
        super().__init__()
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.ema_model = None

    def on_fit_start(self, trainer, pl_module):
       
        self.ema_model = deepcopy(pl_module.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
       
        self.ema_model.to(pl_module.device)
        

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
        if self.ema_model is None: return
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), pl_module.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
            
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), pl_module.model.buffers()):
                ema_buffer.data.copy_(model_buffer.data)

    def on_validation_start(self, trainer, pl_module):
        
        if self.validate_original_weights or self.ema_model is None: return
        self.original_state_dict = deepcopy(pl_module.model.state_dict())
        pl_module.model.load_state_dict(self.ema_model.state_dict())
        # print("Switched to EMA weights for validation")

    def on_validation_end(self, trainer, pl_module):
        
        if self.validate_original_weights or self.ema_model is None: return
        pl_module.model.load_state_dict(self.original_state_dict)
        del self.original_state_dict
        # print("Restored training weights")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        
        if self.ema_model is not None:
            checkpoint['state_dict_ema'] = self.ema_model.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        
        if 'state_dict_ema' in checkpoint:
            self.ema_model = deepcopy(pl_module.model)
            self.ema_model.load_state_dict(checkpoint['state_dict_ema'])
# ==========================================
