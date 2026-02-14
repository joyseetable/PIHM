# 文件路径: train.py

import os
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
from typing import List

# [修改] 使用你的工厂函数
from src.datasets import create_datamodule 

# 导入 System
from systems.system import DistillSystem
# 导入 Callbacks (假设存在)
from src.utils.logging_callbacks import TextLoggingCallback
# from src.callbacks.schdule import DynamicLossWeightScheduler
from omegaconf import OmegaConf, DictConfig
from src.callbacks.ema import EMACallback
def get_center_path(clip_name):
    # 基础路径
    base_dir = "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results"
    USE_RSI_DATASET = False
    # 根据名称判断
    if "ViT-B" in clip_name:
        if USE_RSI_DATASET:
            return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/RSI_clustering_results/cluster_centers_vitb32_RSI_32.npy"
        # L版本对应 160 类
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"
    elif "ViT-L" in clip_name:
        # B版本对应 20 类 (假设你的B版本文件名为 cluster_centers.npy)
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_vitl14_160.npy"
    else:
        # 默认回退
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"
OmegaConf.register_new_resolver("select_path", get_center_path)

@hydra.main(config_path="configs", config_name="train_new", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # 1. 打印配置 (仅主进程)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("--- Hydra Configuration ---")
        print("--------------------------")

    pl.seed_everything(cfg.seed, workers=True)

    # 2. 创建 DataModule (使用工厂函数)
    # [修改] 这里恢复使用 create_datamodule
    datamodule = create_datamodule(cfg)
    
    # 3. 实例化 System
    system = DistillSystem(cfg)

    # --- 4. 权重加载逻辑 (Finetune / Warm start) ---
    load_ckpt_path = cfg.get("resume_from_checkpoint", None)
    if load_ckpt_path:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"\n--- 💡 手动加载权重 (Strict=False) 从: {load_ckpt_path} ---\n")
        
        # 加载 checkpoint 到 CPU
        checkpoint = torch.load(load_ckpt_path, map_location="cpu")
        
        # 加载权重 (strict=False)
        missing_keys, unexpected_keys = system.load_state_dict(checkpoint["state_dict"], strict=False)
        
        if os.environ.get("LOCAL_RANK", "0") == "0":
             print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # --- 5. 构建 Callbacks ---
    callbacks: List[pl.Callback] = []

    # 5.1 Checkpoint
    checkpoint_callback = ModelCheckpoint(
        save_on_train_epoch_end=False,
        **cfg.checkpoint_callback
    )
    callbacks.append(checkpoint_callback)
    # ==========================================
    # [新增] 5.2 EMA Callback 
    # ==========================================
    # decay 设置为 0.999 是通用做法
    # 如果 cfg 里有定义 cfg.train.ema_decay 可以用 cfg 获取，没有就硬编码
    # ema_decay = cfg.get("ema_decay", 0.999) 
    # ema_callback = EMACallback(decay=ema_decay)
    # callbacks.append(ema_callback)
    # ==========================================
    # 5.2 自定义 Logging (确保文件存在)
    # callbacks.append(TextLoggingCallback()) 
    
    # 5.3 动态 Loss 权重 (确保文件存在)
    # callbacks.append(DynamicLossWeightScheduler(cfg)) 

    # --- 6. Loggers ---
    loggers = [
        TensorBoardLogger(
            save_dir=cfg.trainer.default_root_dir,
            name=cfg.project_name,
            version=cfg.dataset.get("name", "version_0"),
        ),
        CSVLogger(
            save_dir=cfg.trainer.default_root_dir,
            name=cfg.project_name,
            version=cfg.dataset.get("name", "version_0"),
        )
    ]

    # --- 7. DDP Strategy ---
    ddp_strategy = DDPStrategy(find_unused_parameters=True)

    # --- 8. 实例化 Trainer ---
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=loggers,
        callbacks=callbacks, 
        strategy=ddp_strategy,
        accelerator="gpu",
        devices="auto", 
        # gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
        # limit_train_batches=1,  # 设为0，表示训练 0 个 batch
        # limit_val_batches=1.0,  # 验证集跑完
        # [关键] 必须为 False，保护我们在 DataModule 中手写的 DDP Sampler 逻辑
        # use_distributed_sampler=False, 
    )
    
    # --- 9. 开始训练 ---
    if cfg.get("test_only", False):
        trainer.test(system, datamodule=datamodule, ckpt_path=load_ckpt_path)
    else:
        # trainer.validate(system, datamodule=datamodule)
        trainer.fit(system, datamodule=datamodule)
        
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("----- Run Finished -----")

if __name__ == '__main__':
    main()