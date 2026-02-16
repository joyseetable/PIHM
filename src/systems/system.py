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
        
        model_type = config.model.get('type', 'ours')
        
        if model_type == 'baseline_full_finetune':
            if self.global_rank == 0:
                print("🚀 [System] Activate CLIP Full Finetune Baseline Mode")
            self.model = CLIPFullFinetune(config)
        else:
            
            self.model = PrototypeGuidedCLIP(config)
        
        
       
        self._setup_human_readable_logging()
        
        
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
        
        if self.global_rank == 0:
            print("\n" + "="*40)
            print("(Trainable Parameters):")
            print("="*40)
            
            total_params = 0
            trainable_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    # print(f"| Size: {param.size()} | Numel: {param.numel()}")
                    trainable_params += param.numel()
                else:
                    
                    pass
            
            print("-" * 40)
            # print(f"📊 {total_params:,}")
            # print(f"🔥 tranable: {trainable_params:,} ({trainable_params/total_params:.2%})")
            print("="*40 + "\n")
            
            
            if hasattr(self, 'logger_human'):
                self.logger_human.info(f"{trainable_params:,}")
    def _setup_human_readable_logging(self):
        
        
        log_dir = self.config.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
       
        self.logger_human = logging.getLogger("human_readable")
        self.logger_human.setLevel(logging.INFO)
        
        
        if not self.logger_human.handlers:
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
           
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
        
        
        if stage == 'fit' and not self.baseline_validated and self.global_rank == 0:
            self.baseline_validated = True

    def forward(self, batch):
        
        return self.model(batch)

    def on_train_epoch_start(self):
        
        if self.global_rank == 0:
            self.logger_human.info(f" Epoch {self.current_epoch}")       
        self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)

    def on_train_batch_start(self, batch, batch_idx):
        
        
        if batch_idx == 0:
            if batch_idx == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                self.logger_human.info(f"GPU: {gpu_memory:.2f} GB")
                # [新增] 打印当前学习率
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.logger_human.info(f"Current LR: {current_lr:.2e}")
            gpu_memory = torch.cuda.memory_allocated() / 1024**3    # GB
            self.logger_human.info(f"GPU: {gpu_memory:.2f} GB")
        if self.global_rank == 0 and batch_idx % 100 == 0:
            self.logger_human.info(f"Epoch {self.current_epoch} - batch {batch_idx}")

    def on_train_epoch_end(self):
        
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
            
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy_loss  
        outputs = self(batch)
        
        total_loss = outputs["loss"]
        # aux_loss = outputs["aux_loss"]
        aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=self.device))
       
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            
        
        if torch.isnan(total_loss):
            if self.global_rank == 0:
                self.logger_human.warning(f"Loss is NaN at batch {batch_idx}. Returning dummy loss.")
            
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy_loss   
        return total_loss 
    
    def configure_optimizers(self):
        
        decay_params = []
        no_decay_params = []
        prompt_params=[]
        
        no_decay_keywords = ['bias', 'LayerNorm', 'ln_', 'bn_', 'embed', 'centers']

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            
            if any(key in name for key in no_decay_keywords):
                no_decay_params.append(param)
            elif 'prompts' in name:
                prompt_params.append(param)
            else:
                
                decay_params.append(param)

        
        optim_groups = [
            {
                "params": decay_params, 
                "weight_decay": self.config.optimizer.get("weight_decay", 0.01) 
            },
            {
                "params": no_decay_params, 
                "weight_decay": 0.0  
            },
            {
                "params": prompt_params,
                "weight_decay": 0.0001  
            }
        ]

        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.optimizer.get("lr", 1e-4)
        )

        
        
        max_epochs = self.config.trainer.max_epochs
        
        
        warmup_epochs = 0 
        
        if max_epochs <= warmup_epochs:
            
            warmup_epochs = 0
            
        
        if warmup_epochs > 0:
            
            scheduler_warmup = LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            
            
            scheduler_cosine = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=1e-6
            )
            
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs] 
            )
        else:
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=1e-6
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
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
                
            
            if 'images' in batch:
                images = batch['images'].to(self.device)
            else:
                return {}
            
            if 'text_tokens' in batch:
                text_tokens = batch['text_tokens'].to(self.device)
            else:
                return {}

            
            if 'image_id' in batch:
                image_ids = batch['image_id']
            elif 'imageid' in batch:
                image_ids = batch['imageid']
            else:
                
                image_ids = torch.full((len(images),), -1, device=self.device)
            
            
            if not isinstance(image_ids, torch.Tensor):
                image_ids = torch.tensor(image_ids)
            image_ids = image_ids.to(self.device)

            
            if self.is_sanity_check and 'zeroshot' in self.helper_models:
                zeroshot_model = self.helper_models['zeroshot']
                image_features = zeroshot_model.encode_image(images)
                text_features = zeroshot_model.encode_text(text_tokens)
            else: 
                zeroshot_model = self.helper_models['zeroshot']  
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)

            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            if batch_idx == 0 and self.global_rank == 0:
                
                feat_sum = image_features[0].sum().item()
                feat_head = image_features[0, :5].cpu().numpy().tolist()
                # print(f"\nSum: {feat_sum:.6f} | Head: {feat_head}")
        
        batch_output = {
            "image_features": image_features.cpu(), 
            "text_features": text_features.cpu(),
            "image_ids": image_ids.cpu() 
        }
        self.validation_outputs.append(batch_output)
        return {}

    def on_validation_epoch_end(self):
        
        
        if self.trainer.is_global_zero and self.validation_outputs:
            self._compute_validation_metrics_simple()
        else:
            
            self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)
            # self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r1', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r5', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/i2t_r10', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r1', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r5', 0.0, sync_dist=True, on_epoch=True)
            self.log(f'val/t2i_r10', 0.0, sync_dist=True, on_epoch=True)
            
        
        self.validation_outputs.clear()
        
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _compute_validation_metrics_simple(self):
        
        if not self.validation_outputs:
            return
            
        
        all_image_features = torch.cat([output["image_features"] for output in self.validation_outputs]).to(self.device)
        all_text_features = torch.cat([output["text_features"] for output in self.validation_outputs]).to(self.device)
        all_image_ids = torch.cat([output["image_ids"] for output in self.validation_outputs]).to(self.device)
        
        
        has_valid_ids = (all_image_ids >= 0).any()
        
        if has_valid_ids:
            
            if self.global_rank == 0:
                print(f"COCO")
            recalls, num_unique_images = self._calculate_recalls_by_id(
                all_image_features, all_text_features, all_image_ids
            )
        else:
            
            if self.global_rank == 0:
                print(f"Flickr30k")
            recalls, num_unique_images = self._calculate_recalls_sequential(
                all_image_features, all_text_features
            )

        
        self.log(f'val/RSUM', 0.0, prog_bar=True, sync_dist=True, on_epoch=True)
        
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
    
    
    def _calculate_recalls_by_id(self, all_image_features, all_text_features, all_image_ids):
        
        
        image_ids_np = all_image_ids.cpu().numpy()
        
        
        unique_ids, unique_indices = np.unique(image_ids_np, return_index=True)
        num_unique_images = len(unique_ids)
        
        if num_unique_images == 0:
            return {}, 0

        unique_image_features = all_image_features[unique_indices]
        
        
        # [Text_N, Image_M]
        logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
        logits_per_image = logits_per_text.t()

        
        unique_ids_tensor = torch.tensor(unique_ids, device=self.device)
        
        ground_truth_mask = (all_image_ids.unsqueeze(1) == unique_ids_tensor.unsqueeze(0))

        
        t2i_results = self._compute_recall_from_logits(logits_per_text, ground_truth_mask)
        i2t_results = self._compute_recall_from_logits(logits_per_image, ground_truth_mask.t())

        return {**t2i_results, **i2t_results}, num_unique_images
    def _calculate_recalls_sequential(self, all_image_features, all_text_features):
        
        captions_per_image = 5
        total_samples = all_image_features.shape[0]
        
        
        num_images = total_samples // captions_per_image
        if total_samples % captions_per_image != 0:
            limit = num_images * captions_per_image
            all_image_features = all_image_features[:limit]
            all_text_features = all_text_features[:limit]
            
        if num_images == 0:
            return {}, 0
            
       
        unique_image_features = all_image_features[::captions_per_image]
        
        
        logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
        logits_per_image = logits_per_text.t()
        
        
        device = logits_per_text.device
        
        # text_indices: [0, 1, 2, ..., N-1] -> target_img_idx: [0, 0, 0, 0, 0, 1, 1, ...]
        text_indices = torch.arange(total_samples, device=device)
        target_img_indices = text_indices // captions_per_image
        
        # gallery_indices: [0, 1, ..., num_images-1]
        gallery_indices = torch.arange(num_images, device=device)
        
        # Mask: [N, num_images]
        ground_truth_mask = (target_img_indices.unsqueeze(1) == gallery_indices.unsqueeze(0))
        
        
        t2i_results = self._compute_recall_from_logits(logits_per_text, ground_truth_mask)
        i2t_results = self._compute_recall_from_logits(logits_per_image, ground_truth_mask.t())
        
        return {**t2i_results, **i2t_results}, num_images
    def _compute_recall_from_logits(self, logits, ground_truth_mask):
        
        k_list = [1, 5, 10]
        max_k = max(k_list)
        
        
        _, top_indices = logits.topk(max_k, dim=1)
        
        
        extracted_corrects = torch.gather(ground_truth_mask, 1, top_indices)
        
        results = {}
        #
        
        prefix = 'i2t_' if logits.shape[0] < logits.shape[1] else 't2i_'
        
        
        for k in k_list:
            corrects_at_k = extracted_corrects[:, :k].any(dim=1)
            
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