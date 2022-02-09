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

    def forward(self, input):

        batch_size, n, dim = input.shape

        q = self.fc_q(input)  # batch_size,n,dim
        k = self.fc_k(input).view(1, batch_size, n, dim)  # 1,batch_size,n,dim
        v = self.fc_v(input).view(1, batch_size, n, dim)  # 1,batch_size,n,dim

        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # n,batch_size,dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # n,batch_size,dim

        out = (numerator / denominator)  # n,batch_size,dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # batch_size,n,dim

        return out


def run():
    sample = torch.randn(50, 49, 512)
    aft = AttentionFreeTransformer(dim=512, n=49)
    output = aft(sample)
    print(output.shape)


if __name__ == '__main__':
    run()
