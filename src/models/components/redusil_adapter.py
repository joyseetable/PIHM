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
    """
    改进版的 ClipAdapter:
    1. 采用 Bottleneck 结构 (降维 -> 升维)
    2. 强制包含残差连接 (Residual Connection)
    3. 加入 Dropout 防止测试集掉点 (过拟合)
    4. 加入缩放因子 (Scaling) 稳定初始化
    """
    def __init__(self, c_in, bottleneck_dim=None, dropout_rate=0.2):
        super(ClipAdapter, self).__init__()
        
        if bottleneck_dim is None:
            bottleneck_dim = c_in // 4  # 默认降维 4 倍
            
        self.adapter_block = nn.Sequential(
            # 1. 下采样降维
            nn.Linear(c_in, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim) if False else nn.Identity(), # 如果是3D tensor(LND)，BatchNorm比较麻烦，可改LayerNorm
            nn.ReLU(inplace=True),
            
            # 2. Dropout 关键防御：解决你提到的测试集掉点问题
            nn.Dropout(dropout_rate),
            
            # 3. 上采样升维
            nn.Linear(bottleneck_dim, c_in)
        )
        
        # 4. 缩放因子：初始化为一个小值（如 0.1），让模型起步时以原始特征为主
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 初始化权重：让变换层在起步时接近 0
        self._init_weights()

    def _init_weights(self):
        # 使用小的正态分布初始化
        nn.init.normal_(self.adapter_block[0].weight, std=0.02)
        nn.init.zeros_(self.adapter_block[0].bias)
        nn.init.normal_(self.adapter_block[4].weight, std=0.02)
        nn.init.zeros_(self.adapter_block[4].bias)

    def forward(self, x):
        """
        x 形状支持: [Batch, Dim] 或 [Seq, Batch, Dim]
        """
        # 记录原始特征用于残差
        identity = x
        
        # 如果是 [Seq, Batch, Dim]，线性层会自动作用在最后一个维度上
        out = self.adapter_block(x)
        
        # 残差连接: 原始特征 + 缩放后的适配特征
        # 这是保证 SOTA 性能的关键，防止深层网络梯度消失
        return identity + self.scale * out