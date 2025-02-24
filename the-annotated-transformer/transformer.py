import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

##################
# Model Overview #
##################

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(
            self.src_embed(src),
            src_mask
        )

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(
            self.tgt_embed(tgt),
            memory,
            src_mask,
            tgt_mask
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(
            self.encode(src, src_mask),
            src_mask,
            tgt,
            tgt_mask
        )

class Generator(nn.Module):
    def __init__(self, d_model, vocab) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

###################
# Helper Function #
###################

# The encoder and decoder are respectively composed of
# a stack of N = 6 identical layers
def clones(module, N: int):
    # A shallow copy constructs a new compound object
    # and then (to the extent possible) inserts references into it
    # to the objects found in the original.
    #
    # A deep copy constructs a new compound object
    # and then, recursively, inserts copies into it of the objects
    # found in the original.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


###########
# Encoder #
###########

class Encoder(nn.Module):
    # N = 6 in `Attention Is All You Need`
    def __init__(self, layer, N: int) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps: float =1e-6) -> None:
        super().__init__()

        # `a_2` and `b_2` are learnable affine transform parameters
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps: float = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Residual Connection
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout: float) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


###########
# Decoder #
###########

class Decoder(nn.Module):
    def __init__(self, layer, N: int) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout: float) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size: int):
    attn_shape = (1, size, size)

    subsequent_mask = torch.triu( # Returns the uppper triangular
        torch.ones(attn_shape),   # part of a matrix
        diagonal=1
    ).type(
        torch.uint8
    )

    return subsequent_mask == 0




