import torch
import torch.nn as nn

class SharedPromptNetwork(nn.Module):
    def __init__(self, shared_dim=512, visual_dim=768, text_dim=512, prompt_len=10, 
                 visual_layers_to_inject=None, text_layers_to_inject=None):
        
        super().__init__()
        
        
        self.prompt_len = prompt_len
        self.latent_tokens = nn.Parameter(torch.empty(prompt_len, shared_dim))
        
        
        nn.init.normal_(self.latent_tokens, std=0.02)
        
        
        
        
        self.visual_projs = nn.ModuleDict()
        if visual_layers_to_inject is not None:
            for layer_idx in visual_layers_to_inject:
                
                self.visual_projs[str(layer_idx)] = nn.Linear(shared_dim, visual_dim)
        
        
        self.text_projs = nn.ModuleDict()
        if text_layers_to_inject is not None:
            for layer_idx in text_layers_to_inject:
                self.text_projs[str(layer_idx)] = nn.Linear(shared_dim, text_dim)

    def forward(self, batch_size=1):
        
        
        tokens = self.latent_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        
        v_prompts = {}
        t_prompts = {}
        
        
        for layer_idx, proj in self.visual_projs.items():
            
            v_prompts[int(layer_idx)] = proj(tokens)
            
        
        for layer_idx, proj in self.text_projs.items():
            t_prompts[int(layer_idx)] = proj(tokens)
            
        return v_prompts, t_prompts