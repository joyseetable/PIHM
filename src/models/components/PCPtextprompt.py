
import torch
import torch.nn as nn

class LightweightMetaNet(nn.Module):
    def __init__(self, input_dim, output_dim, prompt_len, hidden_dim=64):
        super().__init__()
        self.prompt_len = prompt_len
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_len * output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_out')
        nn.init.zeros_(self.net[0].bias)
        
        
        nn.init.constant_(self.net[2].weight, 0)
        nn.init.constant_(self.net[2].bias, 0)

    def forward(self, context_feat):
        B = context_feat.shape[0]
        out = self.net(context_feat)
        return out.reshape(B, self.prompt_len, self.output_dim)