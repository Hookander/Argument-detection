import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.linear2 = nn.Linear(4*d_model, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model)
        y = self.linear1(x)
        y = self.gelu(y)
        y = self.linear2(y)
        return y