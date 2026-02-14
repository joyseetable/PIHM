import os
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, Callback # [修改] 引入 Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
from typing import List
from copy import deepcopy # [新增] 用于深拷贝模型

# [修改] 使用你的工厂函数
from src.datasets import create_datamodule 
from systems.system import DistillSystem
# 导入 Callbacks (假设存在)
from src.utils.logging_callbacks import TextLoggingCallback
# from src.callbacks.schdule import DynamicLossWeightScheduler
from omegaconf import OmegaConf, DictConfig

# ==========================================
# [新增] EMA Callback 类定义 (直接粘贴在这里)
# ==========================================
class EMACallback(Callback):
    """
    Model Exponential Moving Average.
    在训练时维护一个影子模型，验证时使用影子模型的参数。
    """
    def __init__(self, decay=0.999, validate_original_weights=False):
        super().__init__()
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.ema_model = None

    def on_fit_start(self, trainer, pl_module):
        # 训练开始时，复制一份模型
        self.ema_model = deepcopy(pl_module.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        # 确保设备一致
        self.ema_model.to(pl_module.device)
        if trainer.is_global_zero:
            print(f"--- INFO: EMA 机制已启用 (Decay: {self.decay}) ---")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 每个 Batch 更新 EMA 参数
        if self.ema_model is None: return
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), pl_module.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
            # 更新 buffer (BN层统计量等)
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), pl_module.model.buffers()):
                ema_buffer.data.copy_(model_buffer.data)

    def on_validation_start(self, trainer, pl_module):
        # 验证开始：替换权重
        if self.validate_original_weights or self.ema_model is None: return
        self.original_state_dict = deepcopy(pl_module.model.state_dict())
        pl_module.model.load_state_dict(self.ema_model.state_dict())
        # print("Switched to EMA weights for validation")

    def on_validation_end(self, trainer, pl_module):
        # 验证结束：恢复权重
        if self.validate_original_weights or self.ema_model is None: return
        pl_module.model.load_state_dict(self.original_state_dict)
        del self.original_state_dict
        # print("Restored training weights")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 保存模型时，把 EMA 权重也存进去
        if self.ema_model is not None:
            checkpoint['state_dict_ema'] = self.ema_model.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # 加载模型时，恢复 EMA 状态
        if 'state_dict_ema' in checkpoint:
            self.ema_model = deepcopy(pl_module.model)
            self.ema_model.load_state_dict(checkpoint['state_dict_ema'])
# ==========================================
