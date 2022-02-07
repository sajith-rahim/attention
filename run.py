import torch

from self_attention import ScaledDotSelfAttention


def run():
    attn = ScaledDotSelfAttention(256, 4)

    sample = torch.randn(1, 5, 256)

    output = attn(sample, sample, sample)
    print(output)


if __name__ == '__main__':
    run()
