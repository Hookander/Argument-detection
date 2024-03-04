import torch
import torch.nn as nn
import math
from feed_forward import *
from positional_encoding import *

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

    def forward(self, x, key_padding_mask=None):
        """
        Q: (batch_size, seq_len_q, d_model)
        K: (batch_size, seq_len_k, d_model)
        V: (batch_size, seq_len_v, d_model)
        key_padding_mask: (batch_size, seq_len_k)
        """
        Q = self.q_linear(x) # (batch_size, seq_len_q, d_model)
        K = self.k_linear(x) # (batch_size, seq_len_k, d_model)
        V = self.v_linear(x) # (batch_size, seq_len_v, d_model)

        scores = torch.matmul(Q, K.transpose(1, 2)) # (batch_size, seq_len_q, seq_len_k)
        scores = scores / math.sqrt(self.d_model)

        # On met les scores d'attention des paddings à -inf
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1) == 1, -1e9)

        weights = torch.softmax(scores, dim=-1) # (batch_size, seq_len_q, seq_len_k)
        output = torch.matmul(weights, V) # (batch_size, seq_len_q, d_model)
        return output
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h

        self.heads = [SingleHeadSelfAttention(d_model//h) for _ in range(h)]
        self.linear = nn.Linear(d_model, d_model)

        # lorsque vous avez une liste de couche, il faut transformer cette liste en un module nn.ModuleList, sinon les paramètres de ces couches ne seront pas enregistrés par PyTorch
        # si vous ne faites pas ça, lorsque vous faites model.to("cuda"), les paramètres de ces couches ne seront pas envoyés sur le GPU automatiquement
        self.heads = nn.ModuleList(self.heads)

    def forward(self, x, key_padding_mask):
        # x.shape = (batch_size, seq_len, d_model)
        # key_padding_mask.shape = (batch_size, seq_len_max)


        x_split = torch.split(x, self.d_model // self.h, dim = -1)
        y = [head(x_part, key_padding_mask) for head, x_part in zip(self.heads, x_split)]

        y = torch.cat(y, dim = -1)
        y = self.linear(y)
        return y
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.feedforward = FeedForward(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.d_model = d_model


    def forward(self, inputs, key_padding_mask = None):
        """
        inputs: (batch_size, seq_len, d_model)
        key_padding_mask : (batch_size, seq_len_maxs)
        """

        mha_outputs = self.mha(inputs, key_padding_mask = key_padding_mask) # (batch_size, seq_len, d_model)

        mha_outputs = self.layer_norm(mha_outputs + inputs) # (batch_size, seq_len, d_model)

        encoder_outputs = self.feedforward(mha_outputs) # (batch_size, seq_len, d_model)
        encoder_outputs = self.layer_norm(encoder_outputs + mha_outputs) # (batch_size, seq_len, d_model)
        return encoder_outputs

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.layers_list = [EncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        self.d_model = d_model

        self.layers_list = nn.ModuleList(self.layers_list)

    def forward(self, inputs, key_padding_mask = None):

        outputs = inputs
        #print("inputs", inputs.shape)
        for i in range(len(self.layers_list)):
            outputs = self.layers_list[i](outputs, key_padding_mask)
        
        return outputs

# e = TransformerEncoder(2, 2, 2, 5)