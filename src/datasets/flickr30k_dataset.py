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
    """
    为原型引导CLIP模型优化的数据集
    """
    def __init__(self, config, split: str, clip_preprocess, clip_tokenize):
        super().__init__()
        self.config = config
        self.split = split
        self.clip_preprocess = clip_preprocess
        self.clip_tokenize = clip_tokenize
        
        
        # 1. 加载Flickr30k原始JSON标注文件
        annotations_path = hydra.utils.to_absolute_path(self.config.dataset.get(f"{split}_annotations_path"))
        with open(annotations_path, 'r') as f:
            self.dataset_images = json.load(f)['images']
        self.image_dir_base = self.config.dataset.get("image_dir", "./data/flickr30k/images")
        # 2. 建立(image_idx, caption_idx)索引列表
        self.ids = []
        for i, d in enumerate(self.dataset_images):
            # 每个图像有5个caption
            self.ids.extend([(i, j) for j in range(len(d['sentences']))])
        
        # 3. 统计信息
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"数据集: {split}, 总样本数: {len(self.ids)}")
        
        # 4. 缓存所有图像文件名，方便原型分类器使用
        self.filenames = [img['filename'] for img in self.dataset_images]
        
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> dict:
        image_idx, caption_idx = self.ids[index]
        image_info = self.dataset_images[image_idx]
        caption_info = image_info['sentences'][caption_idx]
        filename = image_info['filename']
        
        # 图像路径
        image_path = os.path.join(self.image_dir_base, filename)
        
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            # 返回一个占位符，在collate_fn中过滤
            return {
                "image": None,
                "caption": "",
                "filename": filename,
                "image_path": image_path
            }
        
        # 1. 文本处理：获取原始caption
        caption = caption_info['raw']
        
        # 2. 图像预处理
        image_tensor = self.clip_preprocess(image_pil)
        
        # 3. 返回数据
        return {
            "image": image_tensor,          # [3, 224, 224]
            "caption": caption,             # 原始文本字符串
            "filename": filename,           # 用于调试
            "image_path": image_path,       # 图像完整路径
            "image_idx": image_idx,         # 用于数据追踪
            "caption_idx": caption_idx      # 用于数据追踪
        }


def create_collate_fn(clip_tokenize):
    """
    创建数据集的collate函数
    """
    def collate_fn(batch):
        # 过滤掉无效样本
        batch = [b for b in batch if b["image"] is not None and b["caption"] is not None]
        
        if len(batch) == 0:
            return None
        
        # 提取图像
        images = torch.stack([b["image"] for b in batch])
        
        # 提取文本并批量tokenize
        captions = [b["caption"] for b in batch]
        text_tokens = clip_tokenize(captions, truncate=True)  # 假设clip_tokenize支持批量处理
        
        # 其他元数据
        filenames = [b["filename"] for b in batch]
        image_paths = [b["image_path"] for b in batch]
        
        return {
            "images": images,                # [batch_size, 3, 224, 224]
            "text_tokens": text_tokens,      # [batch_size, seq_len]
            "filenames": filenames,          # 文件名列表
            "image_paths": image_paths,      # 图像路径列表
            "captions": captions        # 原始文本列表
        }

    return collate_fn


def create_data_loader(config, split: str, batch_size: int, 
                      clip_model_name: str = 'ViT-B/32', 
                      num_workers: int = 4,
                      shuffle: bool = None):
    """
    创建数据加载器
    """
    # 加载CLIP获取 transform (不需要加载权重，但CLIP API比较死板)
    # 建议: device='cpu' 避免占用显存
    # jit=False 保持一致性
    _, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)
    
    # 创建tokenize函数
    tokenizer = clip.tokenize
    
    # 创建数据集
    dataset = PrototypeGuidedCLIPDataset(
        config=config,
        split=split,
        clip_preprocess=clip_preprocess,
        clip_tokenize=tokenizer,
    )
    
    # 设置shuffle: 训练打乱，验证/测试不打乱 (关键!)
    if shuffle is None:
        shuffle = (split == 'train')
    
    # 创建collate函数
    collate_fn = create_collate_fn(tokenizer)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True, # 建议开启，加速 Host 到 Device 传输
        drop_last=(split == 'train')  # 训练时丢弃最后一个不完整的batch，稳定BatchNorm(如果有)
    )
    
    return dataloader