# RoPE相对位置编码RotaryEmbedding
# x_m * W_q 中 x_m是第m行向量， W_q分解成列向量q1, q2, q2后面就不写了。x_m, q1都是hidden_size维度
# x_m * W_q = x_m * [q1, q2] = [x_m * q1, x_m * q2]，可以看做Q = x * W_q中的第m行

# x_n * W_k 中 x_n是第m行向量
# x_n * W_k = x_n * [q1, q2] = [x_n * k1, x_n * k2]，可以看做K= x * W_k中的第n行，即K的转置的第n列

# [x_m * q1, x_m * q2]和[x_n * k1, x_n * k2]做内积就是Q * K_T的第m行第n列。
# RoPE相对位置编码的作用就是希望第m行第n列的计算结果有m-n的信息。

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        """
        RoPE位置编码模块。

        参数：
        - dim (int): 嵌入维度。
        - max_position_embeddings (int): 最大位置嵌入长度，默认为2048。
        - base (float): 基础频率参数，默认为10000。
        - device (torch.device): 计算设备，默认为None。
        - scaling_factor (float): 缩放因子，默认为1.0。
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算逆频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seq_len):
        """
        生成RoPE的cos和sin值。
        参数：
        - seq_len: 序列长度。
        返回：
        - cos: 余弦值。
        - sin: 正弦值。
        """
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    
def rotate_half(x):
    """
    将输入张量的后半部分旋转到前半部分。 
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape]