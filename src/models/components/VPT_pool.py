# 文件路径: src/models/components/VPT_pool.py (假设路径)

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelPromptPool(nn.Module):
    """
    基于Gumbel Softmax的提示池
    """
    def __init__(self, pool_size: int, prompt_length: int, 
                 embed_dim: int, top_k: int, embedding_key: str = 'cls'):
        super().__init__()
        
        self.pool_size = pool_size
        self.length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.embedding_key = embedding_key
        
        # 提示池: [Pool_Size, Length, Dim]
        self.prompt = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))
        nn.init.normal_(self.prompt, std=0.02)
        
        # 提示键 (用于匹配): [Pool_Size, Dim]
        self.prompt_key = nn.Parameter(torch.empty(pool_size, embed_dim))
        nn.init.normal_(self.prompt_key, std=0.02)
    
    def l2_normalize(self, x: torch.Tensor, dim: int = -1, epsilon: float = 1e-12) -> torch.Tensor:
        """L2归一化，使用常数 epsilon 避免除零"""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed: torch.Tensor, cls_features: torch.Tensor = None) -> Dict:
        """
        Args:
            x_embed: [Batch, Seq_Len, Dim] (如果key是mean/max则需要)
            cls_features: [Batch, Dim] (如果key是cls则需要)
        """
        out = {}
        
        # 1. 获取查询特征 [Batch, Dim]
        if self.embedding_key == "mean":
            # x_embed: [Batch, Seq_Len, Dim] -> Mean over dim 1
            query_features = torch.mean(x_embed, dim=1)
        elif self.embedding_key == "max":
            query_features = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == "cls":
            if cls_features is None:
                raise ValueError("embedding_key is 'cls' but cls_features is None")
            query_features = cls_features
        else:
            raise ValueError(f"Unknown embedding_key: {self.embedding_key}")
        
        # 2. 归一化
        prompt_key_norm = self.l2_normalize(self.prompt_key, dim=1) # [Pool, Dim]
        query_norm = self.l2_normalize(query_features, dim=1)       # [Batch, Dim]
        
        # 3. 计算相似度 (Cosine Similarity)
        # [Batch, Dim] @ [Dim, Pool] -> [Batch, Pool]
        similarity = torch.matmul(query_norm, prompt_key_norm.t())
        
        # 4. 使用Gumbel Softmax迭代选择 Top-K
        selected_prompts = []
        # selection_logits = [] # 如果需要分析选择分布可以保留
        
        # 复制一份用于迭代掩码
        current_similarity = similarity.clone()
        
        for _ in range(self.top_k):
            # 训练时使用较大的 temperature (0.1-1.0)，推理时使用极小值接近 argmax
            effect_tau = 1.0 if self.training else 0.1 
            # 注意: 如果 tau 太小，梯度会变得非常大(接近NaN)；如果太大，分布太均匀。通常 0.1~1.0 是安全区。
            
            # Gumbel Softmax
            # weights: [Batch, Pool_Size] (One-hot ish if hard=True)
            weights = F.gumbel_softmax(current_similarity, tau=effect_tau, hard=True, dim=-1)
            
            # 选择提示
            # weights: [Batch, Pool, 1, 1]
            # prompt:  [1, Pool, Length, Dim]
            # sum dim 1 -> [Batch, Length, Dim]
            a_selected_prompt = torch.sum(
                weights.unsqueeze(-1).unsqueeze(-1) * self.prompt.unsqueeze(0),
                dim=1
            )
            
            selected_prompts.append(a_selected_prompt)
            # selection_logits.append(weights)
            
            # 屏蔽已选提示
            if self.training:
                # 训练时：软屏蔽，减去一个大数，让它在下一次 softmax 中概率接近 0
                # 这样可以保持梯度的连续性
                current_similarity = current_similarity - weights * 1000
            else:
                # 推理时：直接置为负无穷
                current_similarity = current_similarity.masked_fill(
                    weights.bool(), float('-inf')
                )
        
        # 5. 拼接选中的提示
        # result: [Batch, Top_K * Length, Dim]
        batched_prompt = torch.cat(selected_prompts, dim=1)
        
        # out["similarity"] = similarity # 原始相似度
        out["batched_prompt"] = batched_prompt
        out["orth_loss"] = 0
        
        return out