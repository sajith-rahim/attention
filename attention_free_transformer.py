import torch
from torch import nn


class AttentionFreeTransformer(nn.Module):
    r"""
    Attention Free Transformer.
    This paper replaces the self-attention layer with a new efficient operation,
    that has memory complexity of O(Td), where T is the sequence length and
    d is the dimensionality of embeddings.

    Check: AFT-local and AFT-conv.
    """

    def __init__(self, dim, n=49, pos_bias=False):
        r"""
        :param dim:  dimensionality of embedding / the number of features in the query , key and value vectors.
        :param n: sequnce length
        :param pos_bias: pair-wise positional biases nxn
        """

        super(AttentionFreeTransformer, self).__init__()

        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)

        if pos_bias:
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))

        self.dim = dim
        self.n = n
        self.sigmoid = nn.Sigmoid()

