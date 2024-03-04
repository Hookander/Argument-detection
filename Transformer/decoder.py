import torch
import torch.nn as nn
import math
from feed_forward import *
from positional_encoding import *

class SingleHeadCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_output, key_padding_mask = None):
        """
        decoder_output: (batch_size, seq_len_x, d_model)
        encoder_output: (batch_size, seq_len_y, d_model)
        """
        q = self.q_linear(decoder_output)
        k = self.k_linear(encoder_output)
        v = self.v_linear(encoder_output)

        scores = torch.matmul(q, k.transpose(1, 2)) # (batch_size, seq_len_q, seq_len_k)
        scores = scores / math.sqrt(self.d_model)

        # On met les scores d'attention des paddings à -inf
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1) == 1, -1e9)

        scores = torch.softmax(scores, dim=-1) # (batch_size, seq_len_q, seq_len_k)
        output = torch.matmul(scores, v) # (batch_size, seq_len_q, d_model)
        
        return output

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.attention_heads = nn.ModuleList([SingleHeadCrossAttention(d_model//n_heads) for _ in range(n_heads)])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, decoder_input, encoder_output, key_padding_mask):
        """
        decoder_output: (batch_size, seq_len_x, d_model)
        encoder_output: (batch_size, seq_len_y, d_model)
        key_padding_mask : (batch_size, seq_len_max)
        BM

        faut split les inputs sinon prblm de taille avec les heads ? -> pas fait dans le truc de base
        """

        decoder_output_split = torch.split(decoder_input, self.d_model // self.n_heads, dim = 2)
        encoder_output_split = torch.split(encoder_output, self.d_model // self.n_heads, dim = 2)

        heads = []

        for i in range(self.n_heads):

            heads.append(self.attention_heads[i](decoder_output_split[i], encoder_output_split[i], key_padding_mask = key_padding_mask))


        heads = torch.cat(heads, dim=-1)
        output = self.linear(heads)

        return output


def get_causal_mask(seq_len):
    """
    seq_len: int
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask==1, float('-inf'))
    return mask


class MaskedSingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        seq_len = x.shape[1]
        causal_mask = get_causal_mask(seq_len) # (seq_len, seq_len)
        k_transpose = torch.transpose(k, -2, -1)
        
        scores = self.softmax(torch.matmul(q, k_transpose)/(self.d_model)**(1/2) + causal_mask)

        return torch.matmul(scores, v)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.heads = [MaskedSingleHeadAttention(d_model//h) for _ in range(h)]
        self.linear = nn.Linear(d_model, d_model)

        # lorsque vous avez une liste de couche, il faut transformer cette liste en un module nn.ModuleList, sinon les paramètres de ces couches ne seront pas enregistrés par PyTorch
        # si vous ne faites pas ça, lorsque vous faites model.to("cuda"), les paramètres de ces couches ne seront pas envoyés sur le GPU automatiquement
        self.heads = nn.ModuleList(self.heads)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model)
        x_split = torch.split(x, self.d_model//self.h, dim = -1)
        y = []
        for i in range(len(x_split)):
          y.append(self.heads[i](x_split[i]))

        y = torch.cat(y, dim = -1)
        y = self.linear(y)
        return y

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        
        self.masked_multi_head_attention = MaskedMultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.multi_head_cross_attention = MultiHeadCrossAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)


    def forward(self, decoder_output, encoder_output, key_padding_mask = None):
        """
        decoder_output: (batch_size, seq_len_x, d_model)
        encoder_output: (batch_size, seq_len_y, d_model)
        """

        mha_output = self.masked_multi_head_attention(decoder_output)
        dec_output = self.layer_norm1(decoder_output + mha_output)

        cross_output = self.multi_head_cross_attention(dec_output, encoder_output, key_padding_mask = key_padding_mask)
        cross_output = self.layer_norm2(cross_output)

        output = self.feed_forward(cross_output)
        output = self.layer_norm3(output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):

        super().__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, decoder_input, encoder_output, key_padding_mask = None):

        """
        decoder_output: (batch_size, seq_len_x, d_model)
        encoder_output: (batch_size, seq_len_y, d_model)
        """
        decoder_output = decoder_input
        for i in range(self.n_layers):
          decoder_output = self.decoder_layers[i](decoder_input, encoder_output, key_padding_mask = key_padding_mask)

        return decoder_output
