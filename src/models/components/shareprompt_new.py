
import torch
import torch.nn as nn

class SharedPromptNetwork(nn.Module):
    def __init__(self, 
                 shared_dim=512, 
                 visual_dim=768, 
                 text_dim=512, 
                 prompt_len=10, 
                 visual_layer_mapping=None, 
                 text_layer_mapping=None,
                 dropout=0.1): 

        super().__init__()
        
        self.prompt_len = prompt_len
        self.v_layers = list(visual_layer_mapping.keys()) if visual_layer_mapping else []
        self.t_layers = list(text_layer_mapping.keys()) if text_layer_mapping else []
        
        
        max_v = max(self.v_layers) if self.v_layers else 0
        max_t = max(self.t_layers) if self.t_layers else 0
        max_layers = max(max_v, max_t) + 1


        self.latent_tokens = nn.Parameter(torch.empty(prompt_len, shared_dim))
        nn.init.normal_(self.latent_tokens, std=0.02)
        
        
        self.shared_layer_embeds = nn.Parameter(torch.empty(max_layers, shared_dim))
        nn.init.normal_(self.shared_layer_embeds, std=0.02)

        
        self.v_global_proj = nn.Sequential(
            nn.Linear(shared_dim, visual_dim),
            # nn.Dropout(dropout)
        )
        
        self.t_global_proj = nn.Sequential(
            nn.Linear(shared_dim, text_dim),
            # nn.Dropout(dropout)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_size=1):
        # [Len, Batch, Shared]
        base_tokens = self.latent_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        
        v_prompts = {}
        t_prompts = {}
        
      
        for layer_id in self.v_layers:
            identity = self.shared_layer_embeds[layer_id].reshape(1, 1, -1)
            shared_input = base_tokens + identity
            v_prompts[int(layer_id)] = self.v_global_proj(shared_input)

        
        for layer_id in self.t_layers:
            identity = self.shared_layer_embeds[layer_id].reshape(1, 1, -1)
            shared_input = base_tokens + identity
            t_prompts[int(layer_id)] = self.t_global_proj(shared_input)
            
        return v_prompts, t_prompts