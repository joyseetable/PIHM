import torch
import torch.nn as nn

class PrototypePromptGenerator(nn.Module):

    
    def __init__(self, image_dim, proto_dim, prompt_len, output_dim, hidden_dim=512):
        super().__init__()
        self.prompt_len = prompt_len
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            
            nn.Linear(image_dim + proto_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_len * output_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image_features, prototype_vectors):
        # image_features: [Batch, 768]
        # prototype_vectors: [Batch, 512]
        
       
        combined = torch.cat([image_features, prototype_vectors], dim=-1)
        
        out = self.net(combined)
        return out.reshape(-1, self.prompt_len, self.output_dim)