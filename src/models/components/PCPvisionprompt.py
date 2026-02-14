# 文件路径: src/models/components/PCP_VisualPrompt.py

import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul

class PCPVisualPromptGenerator(nn.Module):
    """
    借鉴 PCP (Preheating with Contextual Prompts) 思想的生成器
    结构：Random Prompt + Prior Prompt (由原型+图像生成) -> 门控融合
    """
    def __init__(self, visual_dim, proto_dim, prompt_len):
        super().__init__()
        self.prompt_len = prompt_len
        self.visual_dim = visual_dim # e.g., 768
        
        # 1. Random Prompt (对应 PCP 中的 visual_prompt_embeddings)
        # 这是一个可学习的静态基底，保证初始稳定性
        self.random_prompt_embeddings = nn.Parameter(torch.zeros(1, prompt_len, visual_dim))
        
        # 2. Prior Prompt 生成层 (替代 PCP 中的 cluster_model)
        # 将 [Image(768) + Proto(512)] 映射到 [768]
        self.prior_projector = nn.Sequential(
            nn.Linear(visual_dim + proto_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim) 
        )
        
        # 3. 投影层 (对应 PCP 中的 visual_prompt_proj)
        self.random_prompt_proj = nn.Linear(visual_dim, visual_dim)
        
        # 4. 融合后的投影层 (对应 PCP 中的 visual_prompt_proj_gather)
        self.final_proj = nn.Linear(visual_dim, visual_dim)
        
        # 5. 门控系数映射 (对应 PCP 中的 cluster_mapping)
        # 用于计算融合比例 alpha
        self.gate_mapping = nn.Linear(visual_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        self._init_pcp_weights()

    def _init_pcp_weights(self):
        # 1. Kaiming Normal 初始化 Linear 层
        for m in [self.random_prompt_proj, self.final_proj, self.gate_mapping]:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # 2. Random Prompt 初始化 (参考 PCP: uniform)
        # visual_val 计算公式参考源码
        val = math.sqrt(6. / float(3 * self.visual_dim)) 
        nn.init.uniform_(self.random_prompt_embeddings.data, -val, val)
        
        # 3. Prior Projector 初始化
        for m in self.prior_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img_feat, proto_feat):
        """
        img_feat: [Batch, 768]
        proto_feat: [Batch, 512]
        """
        B = img_feat.shape[0]
        
        # --- A. 计算 Random Prompt (静态基底) ---
        # [1, L, D] -> [B, L, D]
        # 对应: dropout(proj(embeddings))
        visual_random_prompt = self.dropout(
            self.random_prompt_proj(self.random_prompt_embeddings).expand(B, -1, -1)
        )
        
        # --- B. 计算 Prior Prompt (动态部分) ---
        # 结合 图像 和 原型 -> [Batch, 1280]
        combined_feat = torch.cat([img_feat, proto_feat], dim=-1)
        
        # 生成 Prior 特征: [Batch, D]
        prior_feat = self.prior_projector(combined_feat)
        
        # 扩展到 Prompt 长度: [Batch, L, D]
        visual_prior_prompt = prior_feat.unsqueeze(1).repeat(1, self.prompt_len, 1)
        
        # --- C. 计算融合比例 (Gate) ---
        # 对应: clip(mapping(prior), 0, 1)
        # 这里用 sigmoid 代替 clip 以获得更平滑的梯度
        visual_ratio = torch.sigmoid(self.gate_mapping(visual_prior_prompt))
        
        # --- D. 融合 (Fusion) ---
        # 对应: (1 - ratio) * prior + ratio * random
        # 注意：PCP 源码里把 ratio 给了 random，我们也保持一致
        visual_prompt = (1 - visual_ratio) * visual_prior_prompt + visual_ratio * visual_random_prompt
        
        # --- E. 最终投影 ---
        visual_prompt = self.dropout(self.final_proj(visual_prompt))
        
        return visual_prompt