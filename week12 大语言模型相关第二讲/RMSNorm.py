import torch
import torch.nn as nn

# MLA , deepseek MOE模型， RMSNorm代替LayerNorm，计算简单，仅计算平方均值的根，数值稳定性好，避免精度问题
# 在BF16等低精度训练场景表现出色，适用于低精度运算。所以适用于大规模模型，用于加速训练和推理。
# 激活函数除了sigmond，softmax，relu，gelu，tanh，swish函数（x * sigmond），swish避免了relu梯度为0的问题，对输入进行加权，无解性避免梯度为0
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-8):
        super(RMSNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        return self.gamma * (x / rms)
