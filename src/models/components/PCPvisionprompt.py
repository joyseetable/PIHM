# 文件路径: src/models/components/PCP_VisualPrompt.py

import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul

class PCPVisualPromptGenerator(nn.Module):
   
    def __init__(self, visual_dim, proto_dim, prompt_len):
        super().__init__()
        self.prompt_len = prompt_len
        self.visual_dim = visual_dim # e.g., 768
        
        
        self.random_prompt_embeddings = nn.Parameter(torch.zeros(1, prompt_len, visual_dim))
        
        
        self.prior_projector = nn.Sequential(
            nn.Linear(visual_dim + proto_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim) 
        )
        
        
        self.random_prompt_proj = nn.Linear(visual_dim, visual_dim)
        
        
        self.final_proj = nn.Linear(visual_dim, visual_dim)
        
        
        self.gate_mapping = nn.Linear(visual_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        self._init_pcp_weights()

    def _init_pcp_weights(self):
        
        for m in [self.random_prompt_proj, self.final_proj, self.gate_mapping]:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        
        val = math.sqrt(6. / float(3 * self.visual_dim)) 
        nn.init.uniform_(self.random_prompt_embeddings.data, -val, val)
        
        
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
        
        
        visual_random_prompt = self.dropout(
            self.random_prompt_proj(self.random_prompt_embeddings).expand(B, -1, -1)
        )
        
        
        combined_feat = torch.cat([img_feat, proto_feat], dim=-1)
        
        
        prior_feat = self.prior_projector(combined_feat)
        
        
        visual_prior_prompt = prior_feat.unsqueeze(1).repeat(1, self.prompt_len, 1)
        
        
        visual_ratio = torch.sigmoid(self.gate_mapping(visual_prior_prompt))
        
        
        visual_prompt = (1 - visual_ratio) * visual_prior_prompt + visual_ratio * visual_random_prompt
        
        
        visual_prompt = self.dropout(self.final_proj(visual_prompt))
        
        return visual_prompt