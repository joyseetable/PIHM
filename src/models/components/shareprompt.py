import torch
import torch.nn as nn

class SharedPromptNetwork(nn.Module):
    def __init__(self, shared_dim=512, visual_dim=768, text_dim=512, prompt_len=10, 
                 visual_layers_to_inject=None, text_layers_to_inject=None):
        """
        MMRL 风格的共享提示网络: Latent Token -> Layer-wise Projection
        """
        super().__init__()
        
        # 1. 核心: 模态无关的潜在 Tokens (The "Brain")
        # 形状: [Prompt_Len, Shared_Dim]
        self.prompt_len = prompt_len
        self.latent_tokens = nn.Parameter(torch.empty(prompt_len, shared_dim))
        
        # 初始化: 正态分布 (参考 MMRL)
        nn.init.normal_(self.latent_tokens, std=0.02)
        
        # 2. 投影层 (The "Translators")
        # 我们需要知道哪些层需要注入提示，以便建立对应的投影层
        
        # 视觉端投影层
        self.visual_projs = nn.ModuleDict()
        if visual_layers_to_inject is not None:
            for layer_idx in visual_layers_to_inject:
                # 每一层都有一个独立的 Linear，把 Shared_Dim 映射到 Visual_Dim
                self.visual_projs[str(layer_idx)] = nn.Linear(shared_dim, visual_dim)
        
        # 文本端投影层
        self.text_projs = nn.ModuleDict()
        if text_layers_to_inject is not None:
            for layer_idx in text_layers_to_inject:
                self.text_projs[str(layer_idx)] = nn.Linear(shared_dim, text_dim)

    def forward(self, batch_size=1):
        """
        返回: 
            v_prompts: Dict {layer_idx: [Prompt_Len, Batch, Visual_Dim]}
            t_prompts: Dict {layer_idx: [Prompt_Len, Batch, Text_Dim]}
        """
        # 1. 扩展 Latent Tokens 到 Batch 维度
        # [Len, Shared] -> [Len, Batch, Shared]
        tokens = self.latent_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        
        v_prompts = {}
        t_prompts = {}
        
        # 2. 视觉投影 (Layer-wise)
        for layer_idx, proj in self.visual_projs.items():
            # [Len, Batch, Shared] -> [Len, Batch, Visual_Dim]
            # 投影后再 Permute 成 [Batch, Len, Dim] 或者保持 [Len, Batch, Dim] 取决于你的主模型需求
            # 这里我们输出 [Len, Batch, Dim] 方便直接拼接
            v_prompts[int(layer_idx)] = proj(tokens)
            
        # 3. 文本投影 (Layer-wise)
        for layer_idx, proj in self.text_projs.items():
            t_prompts[int(layer_idx)] = proj(tokens)
            
        return v_prompts, t_prompts