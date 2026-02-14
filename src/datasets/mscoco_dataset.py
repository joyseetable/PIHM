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
    适配扁平化 COCO 2017 JSON 的数据集类
    结构示例:
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
        
        # 1. 加载扁平化的 JSON 标注文件
        # 请确保 config 中对应的路径是指向那个 "coco_flattened.json"
        annotations_path = hydra.utils.to_absolute_path(self.config.dataset.get(f"{split}_annotations_path"))
        
        if not os.path.exists(annotations_path):
             raise FileNotFoundError(f"标注文件未找到: {annotations_path}")

        print(f"正在加载标注文件: {annotations_path} ...")
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # 设置图片文件夹路径
        if split == "train":
            # 默认路径修改为你的实际路径，例如 '.../train2017'
            self.image_dir_base = self.config.dataset.get("image_dir_base_train", "/data/clc/data/mscoco/train2017")
        else:
            # 默认路径修改为 '.../val2017'
            self.image_dir_base = self.config.dataset.get("image_dir_base_val", "/data/clc/data/mscoco/val2017")

        # 2. 统计信息
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"数据集: {split}, 加载样本数: {len(self.annotations)}")
        
        # (可选) 缓存文件名列表，如果外部有用到的话
        # 注意：这里是展平的，会有重复的文件名（因为一张图对应5个caption）
        # self.filenames = [item['filename'] for item in self.annotations]
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index: int) -> dict:
        # 直接获取图文对数据
        ann = self.annotations[index]
        
        filename = ann['filename']
        caption = ann['caption']
        image_id = ann.get('imageid', 0) # 获取imageid，如果没有则默认为0
        
        # 拼接图片完整路径
        image_path = os.path.join(self.image_dir_base, filename)
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            # 打印错误但不要中断训练，返回None由collate_fn过滤
            print(f"无法加载图像 {image_path}: {e}")
            return {
                "image": None,
                "caption": "",
                "filename": filename,
                "image_path": image_path
            }
        
        # 1. 图像预处理 (CLIP Transform)
        image_tensor = self.clip_preprocess(image_pil)
        
        # 2. 返回数据
        return {
            "image": image_tensor,          # [3, 224, 224]
            "caption": caption,             # 原始文本字符串
            "filename": filename,           # 文件名
            "image_path": image_path,       # 完整路径
            "image_id": image_id,           # 图片ID
            "index": index                  # 数据索引
        }

def create_collate_fn(clip_tokenize):
    """
    创建数据集的collate函数
    修正版：包含 image_id 的处理
    """
    def collate_fn(batch):
        # 1. 过滤掉加载失败的样本 (image is None)
        batch = [b for b in batch if b["image"] is not None and b["caption"] is not None]
        
        if len(batch) == 0:
            return None
        
        # 2. 提取图像并堆叠
        images = torch.stack([b["image"] for b in batch])
        
        # 3. 提取文本并批量tokenize
        captions = [b["caption"] for b in batch]
        text_tokens = clip_tokenize(captions, truncate=True)
        
        # 4. 提取元数据
        filenames = [b["filename"] for b in batch]
        image_paths = [b["image_path"] for b in batch]
        
        # -----------------------------------------------------------
        # [修正] 提取 image_id 并转换为 Tensor
        # -----------------------------------------------------------
        # 你的 Dataset __getitem__ 返回的是 "image_id" (int)
        # 我们需要把它变成一个 LongTensor，方便后续在 GPU 上进行 mask 计算
        image_ids = [b["image_id"] for b in batch]
        image_ids_tensor = torch.tensor(image_ids, dtype=torch.long)
        
        return {
            "images": images,                # [batch_size, 3, 224, 224]
            "text_tokens": text_tokens,      # [batch_size, seq_len]
            "filenames": filenames,          # list of strings
            "image_paths": image_paths,      # list of strings
            "captions": captions,            # list of strings
            "image_id": image_ids_tensor     # [batch_size] (Tensor) <--- 新增
        }
    
    return collate_fn

# create_data_loader 函数无需修改，保持原样即可
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