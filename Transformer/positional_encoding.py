import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape = (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model) # shape = (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1)

        # maintenant, on veut faire self.pe = pe
        # mais si on fait ça, lorsqu'on fait model.to("cuda"), self.pe ne sera pas envoyé sur le GPU
        # en utilisant register_buffer, on aura self.pe = pe et self.pe sera envoyé sur le GPU lorsqu'on fait model.to("cuda")
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        #print('pe', self.pe.shape)
        x = x + self.pe[:, :seq_len] #.unsqueeze(0)
        #print("pos_enc", x.shape)
        return x