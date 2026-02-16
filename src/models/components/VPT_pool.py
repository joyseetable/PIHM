# 文件路径: src/models/components/VPT_pool.py (假设路径)

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelPromptPool(nn.Module):

    def __init__(self, pool_size: int, prompt_length: int, 
                 embed_dim: int, top_k: int, embedding_key: str = 'cls'):
        super().__init__()
        
        self.pool_size = pool_size
        self.length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.embedding_key = embedding_key
        
        
        self.prompt = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))
        nn.init.normal_(self.prompt, std=0.02)
        
        
        self.prompt_key = nn.Parameter(torch.empty(pool_size, embed_dim))
        nn.init.normal_(self.prompt_key, std=0.02)
    
    def l2_normalize(self, x: torch.Tensor, dim: int = -1, epsilon: float = 1e-12) -> torch.Tensor:
        
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed: torch.Tensor, cls_features: torch.Tensor = None) -> Dict:
        
        out = {}
        
        
        if self.embedding_key == "mean":
            
            query_features = torch.mean(x_embed, dim=1)
        elif self.embedding_key == "max":
            query_features = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == "cls":
            if cls_features is None:
                raise ValueError("embedding_key is 'cls' but cls_features is None")
            query_features = cls_features
        else:
            raise ValueError(f"Unknown embedding_key: {self.embedding_key}")
        
        
        prompt_key_norm = self.l2_normalize(self.prompt_key, dim=1) # [Pool, Dim]
        query_norm = self.l2_normalize(query_features, dim=1)       # [Batch, Dim]
        
        
        # [Batch, Dim] @ [Dim, Pool] -> [Batch, Pool]
        similarity = torch.matmul(query_norm, prompt_key_norm.t())
        
        
        selected_prompts = []
        # selection_logits = [] 
        
        
        current_similarity = similarity.clone()
        
        for _ in range(self.top_k):
            
            effect_tau = 1.0 if self.training else 0.1 
            
            
            # Gumbel Softmax
            # weights: [Batch, Pool_Size] (One-hot ish if hard=True)
            weights = F.gumbel_softmax(current_similarity, tau=effect_tau, hard=True, dim=-1)
            
           
            # weights: [Batch, Pool, 1, 1]
            # prompt:  [1, Pool, Length, Dim]
            # sum dim 1 -> [Batch, Length, Dim]
            a_selected_prompt = torch.sum(
                weights.unsqueeze(-1).unsqueeze(-1) * self.prompt.unsqueeze(0),
                dim=1
            )
            
            selected_prompts.append(a_selected_prompt)
            # selection_logits.append(weights)
            
            
            if self.training:
                
                current_similarity = current_similarity - weights * 1000
            else:
                #
                current_similarity = current_similarity.masked_fill(
                    weights.bool(), float('-inf')
                )
        
        
        # result: [Batch, Top_K * Length, Dim]
        batched_prompt = torch.cat(selected_prompts, dim=1)
        
        # out["similarity"] = similarity 
        out["batched_prompt"] = batched_prompt
        out["orth_loss"] = 0
        
        return out