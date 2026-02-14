import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import clip
import hydra
import logging
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from models.model import PrototypeGuidedCLIP, CLIPFullFinetune
from src.utils.metrics import info_nce_loss, triplet_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class DistillSystem(pl.LightningModule):
    def __init__(self, config: DictConfig):
        # print("DEBUG: DistillSystem init called with config!", config) 
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        # 使用两个提示池+类别提示的模型
        # 读取配置中的 model.type，默认为 'ours'
        model_type = config.model.get('type', 'ours')
        
        if model_type == 'baseline_full_finetune':
            if self.global_rank == 0:
                print("🚀 [System] Activate CLIP Full Finetune Baseline Mode")
            self.model = CLIPFullFinetune(config)
        else:
            # 使用两个提示池+类别提示的模型 (你的原始模型)
            self.model = PrototypeGuidedCLIP(config)
        # self.model = PrototypeGuidedCLIP(config) 
        
        # 设置人类可读的日志
        self._setup_human_readable_logging()
        
        # 辅助模型（用于基线验证）
        self.helper_models = {}
        if not config.get("test_only", False):
            zeroshot_model, _ = clip.load(config.model.original_clip_name, device="cpu", jit=False)
            self.helper_models['zeroshot'] = zeroshot_model.eval().float()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.validation_outputs = []
        self.test_outputs = []
        self.is_sanity_check = True
        self.baseline_validated = False
        self.stage_switch_triggered = False
        self._print_trainable_parameters()
    def _print_trainable_parameters(self):
        """辅助函数：打印所有参与训练的参数"""
        if self.global_rank == 0:
            print("\n" + "="*40)
            print("🔍 正在检查可训练参数 (Trainable Parameters):")
            print("="*40)
            
            total_params = 0
            trainable_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    print(f"✅ [训练中] {name} | Size: {param.size()} | Numel: {param.numel()}")
                    trainable_params += param.numel()
                else:
                    # 如果你想看被冻结的参数，取消下面这行的注释
                    # print(f"❄️ [已冻结] {name}")
                    pass
            
            print("-" * 40)
            print(f"📊 总参数量: {total_params:,}")
            print(f"🔥 可训练参数量: {trainable_params:,} ({trainable_params/total_params:.2%})")
            print("="*40 + "\n")
            
            # 记录到日志文件
            if hasattr(self, 'logger_human'):
                self.logger_human.info(f"可训练参数量: {trainable_params:,}")
    def _setup_human_readable_logging(self):
        """设置人类可读的文本日志"""
        # 创建日志目录
        log_dir = self.config.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名（包含时间戳）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # 配置日志
        self.logger_human = logging.getLogger("human_readable")
        self.logger_human.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not self.logger_human.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger_human.addHandler(file_handler)
            self.logger_human.addHandler(console_handler)
        
        self.logger_human.info(f"Training started at {timestamp}")
        self.logger_human.info(f"Log file: {log_file}")

    def setup(self, stage: str):
        if 'zeroshot' in self.helper_models:
            self.helper_models['zeroshot'] = self.helper_models['zeroshot'].to(self.device)
        
        # 只在主进程运行基线验证
        if stage == 'fit' and not self.baseline_validated and self.global_rank == 0:
            self.baseline_validated = True

    def forward(self, batch):
        """调用两阶段文本增强模型的 forward 方法"""
        return self.model(batch)

    def on_train_epoch_start(self):
        """训练epoch开始时记录"""
        if self.global_rank == 0:
            self.logger_human.info(f"开始两阶段文本增强训练 Epoch {self.current_epoch}")       
        self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)

    def on_train_batch_start(self, batch, batch_idx):
        """训练batch开始时记录进度"""
        
        if batch_idx == 0:
            if batch_idx == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                self.logger_human.info(f"GPU内存使用: {gpu_memory:.2f} GB")
                # [新增] 打印当前学习率
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.logger_human.info(f"Current LR: {current_lr:.2e}")
            gpu_memory = torch.cuda.memory_allocated() / 1024**3    # GB
            self.logger_human.info(f"GPU内存使用: {gpu_memory:.2f} GB")
        if self.global_rank == 0 and batch_idx % 100 == 0:
            self.logger_human.info(f"Epoch {self.current_epoch} - 处理batch {batch_idx}")

    def on_train_epoch_end(self):
        """训练epoch结束时记录人类可读的日志"""
        if self.global_rank == 0:
            train_loss = self.trainer.callback_metrics.get('train/total_loss')
            aux_loss = self.trainer.callback_metrics.get('train/aux_loss')
            if train_loss is not None:
                self.logger_human.info(f"Epoch {self.current_epoch} - Train Loss: {train_loss.item():.4f}")
            if aux_loss is not None:
                self.logger_human.info(f"Epoch {self.current_epoch} - Aux Loss: {aux_loss.item():.4f}")

    def training_step(self, batch, batch_idx):
        
        if batch is None:
            if self.global_rank == 0:
                self.logger_human.warning(f"Batch {batch_idx} is None. Returning dummy loss.")
            # 返回 0.0 的 loss，并确保 requires_grad=True，这样 DDP 认为它是一个合法的 loss
            # 产生的梯度为 0，不会影响模型权重
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy_loss  
        outputs = self(batch)
        # 获取损失 - 适配新模型的输出键名
        total_loss = outputs["loss"]
        # aux_loss = outputs["aux_loss"]
        aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=self.device))
        # 记录损失
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            
        
        if torch.isnan(total_loss):
            if self.global_rank == 0:
                self.logger_human.warning(f"Loss is NaN at batch {batch_idx}. Returning dummy loss.")
            # 同样返回 dummy loss，防止 DDP 崩溃
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy_loss   
        return total_loss 
    
    def configure_optimizers(self):
        # 1. 参数分组逻辑
        decay_params = []
        no_decay_params = []
        prompt_params=[]
        # 这里的白名单包含你希望“不衰减”的参数名字关键字
        # prompt_embeddings, class_embedding, pos_embed 等通常不衰减
        # bias 和 LayerNorm (ln_, bn_) 通常也不衰减
        no_decay_keywords = ['bias', 'LayerNorm', 'ln_', 'bn_', 'embed', 'centers']

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 检查参数名是否包含不衰减的关键字
            if any(key in name for key in no_decay_keywords):
                no_decay_params.append(param)
            elif 'prompts' in name:
                prompt_params.append(param)
            else:
                # 剩下的主要是 Adapter 的 Linear.weight
                decay_params.append(param)

        # 2. 定义参数组
        optim_groups = [
            {
                "params": decay_params, 
                "weight_decay": self.config.optimizer.get("weight_decay", 0.01) # Adapter 使用 0.01
            },
            {
                "params": no_decay_params, 
                "weight_decay": 0.0  # Prompt 和 Bias 不衰减
            },
            {
                "params": prompt_params,
                "weight_decay": 0.0001  # Prompt 不衰减
            }
        ]

        # 3. 创建优化器
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.optimizer.get("lr", 1e-4)
        )

        
        # 获取最大 Epoch 数
        max_epochs = self.config.trainer.max_epochs
        
        # === 设置预热参数 ===
        # 预热 epoch 数，建议设为 5 或者 max_epochs 的 10%
        warmup_epochs = 0 
        
        if max_epochs <= warmup_epochs:
            # 保护措施：如果总 epoch 很短，就别预热了，或者缩短预热
            warmup_epochs = 0
            
        # 2. 定义调度器
        if warmup_epochs > 0:
            # 阶段 A: 线性预热 (Linear Warmup)
            # 从 lr * start_factor 开始，线性增加到 lr * 1.0
            scheduler_warmup = LinearLR(
                optimizer, 
                start_factor=0.1, # 初始学习率为 base_lr * 0.01
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            
            # 阶段 B: 余弦退火 (Cosine Annealing)
            # T_max 是剩余的 epoch 数
            scheduler_cosine = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=1e-6
            )
            
            # 串联调度器
            scheduler = SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs] # 在第 warmup_epochs 个 epoch 切换
            )
        else:
            # 如果不预热，直接使用余弦退火
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=1e-6
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # 每个 Epoch 更新一次 LR
                "frequency": 1,
            }
        }
        


    def on_validation_epoch_start(self):
        self.is_sanity_check = self.trainer.sanity_checking
        self.validation_outputs.clear()

    def validation_step(self, batch, batch_idx):
        # self.model.eval()
        with torch.no_grad():
            if batch is None:
                return {}
                
            # 1. 获取图片和文本
            if 'images' in batch:
                images = batch['images'].to(self.device)
            else:
                return {}
            
            if 'text_tokens' in batch:
                text_tokens = batch['text_tokens'].to(self.device)
            else:
                return {}

            # 2. 尝试获取 image_id (兼容两种情况)
            # COCO Dataset 通常返回 'image_id' 或 'imageid'
            # 如果 Dataset 没有返回 ID，我们创建一个全是 -1 的 Tensor
            if 'image_id' in batch:
                image_ids = batch['image_id']
            elif 'imageid' in batch:
                image_ids = batch['imageid']
            else:
                # 标记为 -1，表示没有 ID，需要在 epoch_end 时使用旧逻辑
                image_ids = torch.full((len(images),), -1, device=self.device)
            
            # 确保 image_ids 是 Tensor 且在正确的设备上
            if not isinstance(image_ids, torch.Tensor):
                image_ids = torch.tensor(image_ids)
            image_ids = image_ids.to(self.device)

            # 3. 模型前向传播 (保持不变)
            if self.is_sanity_check and 'zeroshot' in self.helper_models:
                zeroshot_model = self.helper_models['zeroshot']
                image_features = zeroshot_model.encode_image(images)
                text_features = zeroshot_model.encode_text(text_tokens)
            else: 
                zeroshot_model = self.helper_models['zeroshot']  
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)

            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            if batch_idx == 0 and self.global_rank == 0:
                # 打印第一张图特征的前10位和总和
                feat_sum = image_features[0].sum().item()
                feat_head = image_features[0, :5].cpu().numpy().tolist()
                print(f"\n[指纹-训练态] Sum: {feat_sum:.6f} | Head: {feat_head}")
        # 4. 返回结果
        batch_output = {
            "image_features": image_features.cpu(), 
            "text_features": text_features.cpu(),
            "image_ids": image_ids.cpu() # 传出 ID (可能是真实ID，也可能是 -1)
        }
        self.validation_outputs.append(batch_output)
        return {}

    def on_validation_epoch_end(self):
        """修复多GPU验证同步问题"""
        # 只在主进程计算指标
        if self.trainer.is_global_zero and self.validation_outputs:
            self._compute_validation_metrics_simple()
        else:
            # 为所有进程记录默认值
            self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)
            # self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r1', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r5', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r10', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r1', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r5', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r10', 0.0, sync_dist=True, on_epoch=True)
            print("非主进程把RSUM字符log一下")
        # 所有进程都清空输出
        self.validation_outputs.clear()
        
        # 关键修复：添加同步屏障确保所有进程都完成验证
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _compute_validation_metrics_simple(self):
        """兼容 Flickr30k (顺序) 和 COCO (ID) 的验证指标计算"""
        if not self.validation_outputs:
            return
            
        # 1. 聚合数据
        all_image_features = torch.cat([output["image_features"] for output in self.validation_outputs]).to(self.device)
        all_text_features = torch.cat([output["text_features"] for output in self.validation_outputs]).to(self.device)
        all_image_ids = torch.cat([output["image_ids"] for output in self.validation_outputs]).to(self.device)
        
        # 2. 智能判断使用哪种计算逻辑
        # 检查是否包含有效的 Image ID (即不全为 -1)
        has_valid_ids = (all_image_ids >= 0).any()
        
        if has_valid_ids:
            # === 方案 A: COCO 模式 (基于 ID 匹配) ===
            if self.global_rank == 0:
                print(f"检测到有效 image_id，使用 ID 匹配模式进行评估 (适合 COCO)...")
            recalls, num_unique_images = self._calculate_recalls_by_id(
                all_image_features, all_text_features, all_image_ids
            )
        else:
            # === 方案 B: Flickr30k 旧模式 (基于顺序切片) ===
            if self.global_rank == 0:
                print(f"未检测到 image_id，使用顺序切片模式进行评估 (适合 Flickr30k)...")
            recalls, num_unique_images = self._calculate_recalls_sequential(
                all_image_features, all_text_features
            )

        # 3. 记录日志 (保持不变)
        self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)
        # 占位防止报错
        for k in ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']:
            self.log(f'val/{k}', 0.0, sync_dist=True, on_epoch=True)

        if recalls:
            rsum = sum([recalls['i2t_r1'], recalls['i2t_r5'], recalls['i2t_r10'], 
                        recalls['t2i_r1'], recalls['t2i_r5'], recalls['t2i_r10']])
            
            self.log(f'val/RSUM', rsum, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r1', recalls['i2t_r1'], sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r5', recalls['i2t_r5'], sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r10', recalls['i2t_r10'], sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r1', recalls['t2i_r1'], sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r5', recalls['t2i_r5'], sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r10', recalls['t2i_r10'], sync_dist=True, on_epoch=True)
            
            if self.global_rank == 0:
                self.logger_human.info(f"Epoch {self.current_epoch} - 唯一图片库数量: {num_unique_images} - Validation Results:")
                
                self.logger_human.info(f"  RSUM: {rsum:.2f}")
                self.logger_human.info(f"  Image→Text R@1: {recalls['i2t_r1']:.2f}")
                self.logger_human.info(f"  Image→Text R@5: {recalls['i2t_r5']:.2f}")
                self.logger_human.info(f"  Image→Text R@10: {recalls['i2t_r10']:.2f}")
                self.logger_human.info(f"  Text→Image R@1: {recalls['t2i_r1']:.2f}")
                self.logger_human.info(f"  Text→Image R@5: {recalls['t2i_r5']:.2f}")
                self.logger_human.info(f"  Text→Image R@10: {recalls['t2i_r10']:.2f}")
    
    # def _compute_validation_metrics_simple(self):
    #     """简化版验证指标计算，只使用主进程数据"""
    #     if not self.validation_outputs:
    #         return
            
    #     stage_name = "val_zeroshot" if self.is_sanity_check else "val"
        
    #     all_image_features = torch.cat([output["image_features"] for output in self.validation_outputs]).to(self.device)
    #     all_text_features = torch.cat([output["text_features"] for output in self.validation_outputs]).to(self.device)
        
    #     recalls, num_images = self._calculate_recalls_complete(all_image_features, all_text_features, stage_name)
    #     self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)
    #     self.log(f'val/i2t_r1', 0.0, sync_dist=True, on_epoch=True)
    #     self.log(f'val/i2t_r5', 0.0, sync_dist=True, on_epoch=True)
    #     self.log(f'val/i2t_r10', 0.0, sync_dist=True, on_epoch=True)
    #     self.log(f'val/t2i_r1', 0.0, sync_dist=True, on_epoch=True)
    #     self.log(f'val/t2i_r5', 0.0, sync_dist=True, on_epoch=True)
    #     self.log(f'val/t2i_r10', 0.0, sync_dist=True, on_epoch=True)
    #     if recalls:
    #         rsum = sum([recalls['i2t_r1'], recalls['i2t_r5'], recalls['i2t_r10'], 
    #                     recalls['t2i_r1'], recalls['t2i_r5'], recalls['t2i_r10']])
            
    #         # 记录指标
    #         self.log(f'val/RSUM', rsum, prog_bar=True, sync_dist=True, on_epoch=True)
    #         self.log(f'val/i2t_r1', recalls['i2t_r1'], sync_dist=True, on_epoch=True)
    #         self.log(f'val/i2t_r5', recalls['i2t_r5'], sync_dist=True, on_epoch=True)
    #         self.log(f'val/i2t_r10', recalls['i2t_r10'], sync_dist=True, on_epoch=True)
    #         self.log(f'val/t2i_r1', recalls['t2i_r1'], sync_dist=True, on_epoch=True)
    #         self.log(f'val/t2i_r5', recalls['t2i_r5'], sync_dist=True, on_epoch=True)
    #         self.log(f'val/t2i_r10', recalls['t2i_r10'], sync_dist=True, on_epoch=True)
            
    #         if self.global_rank == 0:
    #             self.logger_human.info(f"Epoch {self.current_epoch} - 图片数量: {num_images} - Validation Results:")
    #             self.logger_human.info(f"  RSUM: {rsum:.2f}")
    #             self.logger_human.info(f"  Image→Text R@1: {recalls['i2t_r1']:.2f}")
    #             self.logger_human.info(f"  Image→Text R@5: {recalls['i2t_r5']:.2f}")
    #             self.logger_human.info(f"  Image→Text R@10: {recalls['i2t_r10']:.2f}")
    #             self.logger_human.info(f"  Text→Image R@1: {recalls['t2i_r1']:.2f}")
    #             self.logger_human.info(f"  Text→Image R@5: {recalls['t2i_r5']:.2f}")
    #             self.logger_human.info(f"  Text→Image R@10: {recalls['t2i_r10']:.2f}")
    def _calculate_recalls_by_id(self, all_image_features, all_text_features, all_image_ids):
        """基于 image_id 的精确匹配 (COCO Style)"""
        # 转为 numpy
        image_ids_np = all_image_ids.cpu().numpy()
        
        # 1. 提取唯一图片
        unique_ids, unique_indices = np.unique(image_ids_np, return_index=True)
        num_unique_images = len(unique_ids)
        
        if num_unique_images == 0:
            return {}, 0

        unique_image_features = all_image_features[unique_indices]
        
        # 2. 计算相似度
        # [Text_N, Image_M]
        logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
        logits_per_image = logits_per_text.t()

        # 3. 生成 Mask
        unique_ids_tensor = torch.tensor(unique_ids, device=self.device)
        # ground_truth[i][j] = True 表示 Text[i] 属于 Image[j]
        ground_truth_mask = (all_image_ids.unsqueeze(1) == unique_ids_tensor.unsqueeze(0))

        # 4. 计算指标
        t2i_results = self._compute_recall_from_logits(logits_per_text, ground_truth_mask)
        i2t_results = self._compute_recall_from_logits(logits_per_image, ground_truth_mask.t())

        return {**t2i_results, **i2t_results}, num_unique_images
    def _calculate_recalls_sequential(self, all_image_features, all_text_features):
        """基于固定顺序 (1图5文) 的匹配 (Flickr30k Old Style)"""
        captions_per_image = 5
        total_samples = all_image_features.shape[0]
        
        # 确保样本数能被 5 整除，否则切掉多余的
        num_images = total_samples // captions_per_image
        if total_samples % captions_per_image != 0:
            limit = num_images * captions_per_image
            all_image_features = all_image_features[:limit]
            all_text_features = all_text_features[:limit]
            
        if num_images == 0:
            return {}, 0
            
        # 1. 提取唯一图片特征 (每隔5个取1个)
        unique_image_features = all_image_features[::captions_per_image]
        
        # 2. 计算相似度 [Text(N), Image(N/5)]
        logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
        logits_per_image = logits_per_text.t()
        
        # 3. 生成 Mask (基于索引关系)
        # 第 i 个文本 对应 第 i // 5 张图
        # 我们手动构造一个 Mask，效果等同于 ID 匹配
        device = logits_per_text.device
        
        # text_indices: [0, 1, 2, ..., N-1] -> target_img_idx: [0, 0, 0, 0, 0, 1, 1, ...]
        text_indices = torch.arange(total_samples, device=device)
        target_img_indices = text_indices // captions_per_image
        
        # gallery_indices: [0, 1, ..., num_images-1]
        gallery_indices = torch.arange(num_images, device=device)
        
        # Mask: [N, num_images]
        ground_truth_mask = (target_img_indices.unsqueeze(1) == gallery_indices.unsqueeze(0))
        
        # 4. 复用同一个计算函数
        t2i_results = self._compute_recall_from_logits(logits_per_text, ground_truth_mask)
        i2t_results = self._compute_recall_from_logits(logits_per_image, ground_truth_mask.t())
        
        return {**t2i_results, **i2t_results}, num_images
    def _compute_recall_from_logits(self, logits, ground_truth_mask):
        """通用向量化 Recall 计算"""
        k_list = [1, 5, 10]
        max_k = max(k_list)
        
        # 取前K名
        _, top_indices = logits.topk(max_k, dim=1)
        
        # 根据 Mask 判断是否正确
        extracted_corrects = torch.gather(ground_truth_mask, 1, top_indices)
        
        results = {}
        # 为了兼容上面的 key 命名，这里稍微改一下返回的 key 格式，让调用者去前缀
        # 实际上我在上面的调用里已经处理了 key 的合并
        # 这里只返回 r1, r5, r10 即可
        
        prefix = 'i2t_' if logits.shape[0] < logits.shape[1] else 't2i_'
        # 注意：这里为了方便，我直接返回不带前缀的 key，由调用者组装，或者直接在这里根据 shape 猜
        # 但最稳妥的是调用者加前缀。为了简单，我让函数返回 r1, r5...
        
        for k in k_list:
            corrects_at_k = extracted_corrects[:, :k].any(dim=1)
            # 乘以 100 转为百分比
            score = corrects_at_k.float().mean().item() * 100
            if logits.shape[0] < logits.shape[1]: # Image Query -> Text Gallery (I2T)
                 results[f'i2t_r{k}'] = score
            else: # Text Query -> Image Gallery (T2I)
                 results[f't2i_r{k}'] = score
                 
        return results
    
    def _calculate_recalls_complete(self, all_image_features, all_text_features, stage: str):
        captions_per_image = 5
        total_image_text_pairs = all_image_features.shape[0]
        num_images = total_image_text_pairs // captions_per_image 
        
        if total_image_text_pairs % captions_per_image != 0:
            adjusted_size = num_images * captions_per_image
            all_image_features = all_image_features[:adjusted_size]
            all_text_features = all_text_features[:adjusted_size]
        
        if num_images == 0: 
            return {}, 0
        
        unique_image_features = all_image_features[::captions_per_image] 
        if unique_image_features.shape[0] != num_images: 
            return {}, 0

        recalls = self._compute_recalls_corrected(unique_image_features, all_text_features, num_images, captions_per_image)
        return recalls, num_images

    def _compute_recalls_corrected(self, unique_image_features, all_text_features, num_images, captions_per_image):
        similarity_i2t = unique_image_features @ all_text_features.t()
        i2t_ranks = []
        for i in range(num_images):
            target_text_indices = list(range(i * captions_per_image, (i + 1) * captions_per_image))
            scores = similarity_i2t[i]
            _, indices = torch.sort(scores, descending=True)
            best_rank = len(scores) + 1
            for target_idx in target_text_indices:
                rank = (indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
                if rank < best_rank:
                    best_rank = rank
            i2t_ranks.append(best_rank)

        similarity_t2i = all_text_features @ unique_image_features.t()
        t2i_ranks = []
        for text_idx in range(num_images * captions_per_image):
            target_image_idx = text_idx // captions_per_image
            scores = similarity_t2i[text_idx]
            _, indices = torch.sort(scores, descending=True)
            rank = (indices == target_image_idx).nonzero(as_tuple=True)[0].item() + 1
            t2i_ranks.append(rank)
        
        i2t_ranks_tensor = torch.tensor(i2t_ranks)
        t2i_ranks_tensor = torch.tensor(t2i_ranks)
        
        return {
            'i2t_r1': (i2t_ranks_tensor <= 1).float().mean().item() * 100,
            'i2t_r5': (i2t_ranks_tensor <= 5).float().mean().item() * 100,
            'i2t_r10': (i2t_ranks_tensor <= 10).float().mean().item() * 100,
            't2i_r1': (t2i_ranks_tensor <= 1).float().mean().item() * 100,
            't2i_r5': (t2i_ranks_tensor <= 5).float().mean().item() * 100,
            't2i_r10': (t2i_ranks_tensor <= 10).float().mean().item() * 100,
        }