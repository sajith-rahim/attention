import torch
import torch.nn as nn

from plot_util import plot


class ScaledDotSelfAttention(nn.Module):
    r"""
    Multi-head self attention.
    * query, key and val has same dimension.
    """

    def __init__(self, dim, n_heads):
        r"""
        :parameter
        :param dim: embedded input dimension (output dim as well)
        :param n_heads: number of heads
        """
        super(ScaledDotSelfAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = dim

        assert dim % n_heads == 0, "Dimension has to divisible by n_heads iorder to split!"
        
        # d_k = d_v
        self.dim = dim//n_heads

        self.Wq = nn.Linear(dim, n_heads * dim)
        self.Wk = nn.Linear(dim, n_heads * dim)
        self.Wv = nn.Linear(dim, n_heads * dim)

        self.out = nn.Linear(n_heads * dim, dim)

    def forward(self, queries, keys, values, batch_size=None):
        r"""

        :param queries: [batch_size, query_len, dim]
        :param keys: [batch_size, key_len, dim]
        :param values: [batch_size, key_len, dim]
        :param batch_size: batch_size
        :return: scaled attn product
        """
        batch_size, query_len = queries.shape[:2]

        key_len = keys.shape[1]

        q = self.Wq(queries)
        k = self.Wk(keys)
        v = self.Wv(values)

        # Slicing: [batch_size, query_len, input_dim] -> [batch_size, query_len, n_heads, dim]
        q = q.view(batch_size, query_len, self.n_heads, self.dim)
        # [batch_size, query_len, n_heads, dim] -> [batch_size,n_heads,query_len,dim]
        q = q.permute(0, 2, 1, 3)

        # Slicing
        k = k.view(batch_size, key_len, self.n_heads, self.dim)
        # [batch_size, key_len, n_heads, dim] -> [batch_size,n_heads,dim,key_len] : K^T
        k = k.permute(0, 2, 3, 1)

        attn = torch.matmul(q, k) / torch.sqrt(torch.tensor(float(self.dim)))  # [batch_size,n_heads,query_len,key_len]

        attn = torch.softmax(attn, -1)

        # plot(attn)

        v = v.view(batch_size, key_len, self.n_heads, self.dim)
        # [batch_size, key_len, n_heads, dim] -> [batch_size,n_heads,key_len,dim]
        v = v.permute(0, 2, 1, 3)

        v_ = torch.matmul(attn, v)
        # [batch_size, query_len, n_heads * dim]
        v_ = v_.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.n_heads * self.dim)

        out = self.out(v_)  # (batch_size, query_len, dim)
        return out
