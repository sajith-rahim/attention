import torch
import torch.nn as nn

import matplotlib.pyplot as plt

def plot(attn):
    r"""
    Utility to print attn
    :param attn:
    :return: None
    """

    total = attn.shape[1]
    Cols = 2

    # Compute rows required

    rows = total // Cols
    rows += total % Cols

    # Create a Position index

    position = range(1, total + 1)

    fig = plt.figure(1)
    for k in range(total):
        # add every single subplot to the figure with a for loop

        ax = fig.add_subplot(rows, Cols, position[k])
        ax.matshow(attn[0][k].detach().numpy())

    plt.show()





class SelfAttention(nn.Module):
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
        super(SelfAttention, self).__init__()

        self.n_heads = n_heads
        self.dim = dim

        assert dim % n_heads == 0, "Dimension has to divisible by n_heads iorder to split!"

        self.Wq = nn.Linear(dim, n_heads * dim)
        self.Wk = nn.Linear(dim, n_heads * dim)
        self.Wv = nn.Linear(dim, n_heads * dim)

        self.out = nn.Linear(n_heads * dim, dim)

    def forward(self, queries, keys, values, batch_size=None):
        r"""

        :param queries: [batch_size, n_queries, dim]
        :param keys: [batch_size, n_keys, dim]
        :param values: [batch_size, n_keys, dim]
        :param batch_size: batch_size
        :return: scaled attn product
        """
        batch_size, n_queries = queries.shape[:2]

        n_keys = keys.shape[1]

        q = self.Wq(queries)
        k = self.Wk(keys)
        v = self.Wv(values)

        # Slicing: [batch_size, n_queries, input_dim] -> [batch_size, n_queries, n_heads, dim]
        q = q.view(batch_size, n_queries, self.n_heads, self.dim)
        # [batch_size, n_queries, n_heads, dim] -> [batch_size,n_heads,n_queries,dim]
        q = q.permute(0, 2, 1, 3)

        # Slicing
        k = k.view(batch_size, n_keys, self.n_heads, self.dim)
        # [batch_size, n_keys, n_heads, dim] -> [batch_size,n_heads,n_keys,dim]
        k = k.permute(0, 2, 3, 1)

        attn = torch.matmul(q, k) / torch.sqrt(torch.tensor(float(self.dim)))  # [batch_size,n_heads,n_queries,n_keys]

        attn = torch.softmax(attn, -1)

        # plot(attn)

        v = v.view(batch_size, n_keys, self.n_heads, self.dim)
        # [batch_size, n_keys, n_heads, dim] -> [batch_size,n_heads,n_keys,dim]
        v = v.permute(0, 2, 1, 3)

        v_ = torch.matmul(attn, v)
        # [batch_size, n_queries, n_heads * dim]
        v_ = v_.permute(0, 2, 1, 3).contiguous().view(batch_size, n_queries, self.n_heads * self.dim)

        out = self.out(v_)  # (batch_size, n_queries, dim)
        return out


def run():
    attn = SelfAttention(256, 4)

    sample = torch.randn(1, 5, 256)

    output = attn(sample, sample, sample)
    print(output)


if __name__ == '__main__':
    run()
