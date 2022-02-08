import torch.nn as nn
import torch.nn.functional as F
import torch

from transformer.Attention import Attention


class TransformerEncoderLayer(nn.Module):
    r"""
    Encoder Layer
    """

    def __init__(self, d_model, n_heads, dim_feedforward=2048, attention_dropout_rate=0.1, projection_dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=n_heads,
                                      attn_dropout_rate=attention_dropout_rate,
                                      projection_dropout_rate=projection_dropout_rate)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(projection_dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(projection_dropout_rate)

        # not using stochastic depth
        # self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        skip = src
        src = self.pre_norm(src)
        src = self.self_attn(src)
        src = self.norm1(src)
        # src = src + skip

        skip = self.linear1(src)
        skip = self.activation(skip)
        skip = self.dropout1(skip)
        skip = self.linear2(skip)
        src = src + self.dropout2(skip)
        return src
