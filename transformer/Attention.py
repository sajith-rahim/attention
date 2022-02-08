import torch.nn as nn
import torch


class Attention(nn.Module):
    r"""
    Scaled Dot Product Attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_dropout_rate=0.1, projection_dropout_rate=0.1):
        super(Attention, self).__init__()

        assert dim % num_heads == 0, "Dimension has to divisible by n_heads inorder to split!"

        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        # combing q,k,v as single layer [d, d*3]
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(attn_dropout_rate)

        self.projection = nn.Linear(dim, dim)
        self.projection_drop = nn.Dropout(projection_dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        # 3 matrices - q,k,v
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = qkv.unbind(0)  # make torch-script happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.projection_drop(x)
        return x


