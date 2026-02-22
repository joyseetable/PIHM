# import torch
# import torch.nn as nn

# class ClipAdapter(nn.Module):
#     def __init__(self, c_in, bottleneck_dim=None, dropout=0.1):
#         super().__init__()
#         if bottleneck_dim is None:
#             bottleneck_dim = c_in // 4
            
#         self.adapter = nn.Sequential(
#             nn.Linear(c_in, bottleneck_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(bottleneck_dim, c_in)
#         )

#     def forward(self, x):
#         return x + self.adapter(x)

import torch
import torch.nn as nn

class ClipAdapter(nn.Module):
    
    def __init__(self, c_in, bottleneck_dim=None, dropout_rate=0.2):
        super(ClipAdapter, self).__init__()
        
        if bottleneck_dim is None:
            bottleneck_dim = c_in // 4  
            
        self.adapter_block = nn.Sequential(
            
            nn.Linear(c_in, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim) if False else nn.Identity(), # 如果是3D tensor(LND)，BatchNorm比较麻烦，可改LayerNorm
            nn.ReLU(inplace=True),
            
            
            nn.Dropout(dropout_rate),
            
            
            nn.Linear(bottleneck_dim, c_in)
        )
        
        
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        
        self._init_weights()

    def _init_weights(self):
        
        nn.init.normal_(self.adapter_block[0].weight, std=0.02)
        nn.init.zeros_(self.adapter_block[0].bias)
        nn.init.normal_(self.adapter_block[4].weight, std=0.02)
        nn.init.zeros_(self.adapter_block[4].bias)

    def forward(self, x):
        identity = x
        out = self.adapter_block(x)
        return identity + self.scale * out