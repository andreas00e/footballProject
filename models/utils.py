import math

import torch
from torchtyping import TensorType
import torch.nn as nn 

class PositionalEncoding(nn.Module):
    # Passed before attention  

    def __init__(self, d_model: int, dropout: int=0.1, max_len: int=5000) -> None:
        """
        Args:
            d_model (int): Dimension of transformer model 
            dropout (int, optional): Dropout probability. Defaults to 0.1.
            max_len (int, optional): Maximum number of tokens in one sequence. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    @property
    def postionalEncoding(self): 
        return self.pe

    def forward(self, x: TensorType["seq_len", "d_model"]) -> TensorType["seq_len", "d_model"]:
        x = x + self.pe[:x.size(0), :] # [seq_len, d_model]
        return self.dropout(x)