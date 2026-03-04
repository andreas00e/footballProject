import math
from typing import List, Union

import torch
import torch.nn as nn 

from torchtyping import TensorType

from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_mean_pool


def list2sequential(features: List) -> List: 
    return [layer for i in range(len(features)-1) for layer in (nn.Linear(features[i], features[i+1]), nn.ReLU())][:-1]
    
class GraphModule(nn.Module): 
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.model = GAT(**args)
    
    def forward(self, x: TensorType["n_nodes", "n_features"], edge_index: TensorType["2", "n_edges"], 
                batch: Union[None | TensorType["1", "num_nodes"]]) -> TensorType["*"]:
        x = self.model(x, edge_index)
        if isinstance(batch, torch.Tensor):
            x = global_mean_pool(x, batch)
        return x   
        
class PlayerEmbeddingMLP(nn.Module): 
    def __init__(self, player_features: List) -> None:
        super().__init__()
        self.model = nn.Sequential(*list2sequential(player_features))
        
    def forward(self, x: TensorType["*"]) -> TensorType["*"]: 
        return self.model(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: int, max_len: int) -> None:
        """
        Args:
            d_model (int): Dimension of transformer model 
            dropout (int, optional): Dropout probability. Defaults to 0.1.
            max_len (int, optional): Maximum number of tokens in one sequence. Defaults to 5000.
        """  
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("_", pe) 
    
    @property
    def postionalEncoding(self): 
        return self.pe

class RMSELoss(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.mse = nn.MSELoss() 
        
    def forward(self, y, y_hat): 
        return torch.sqrt(self.mse(y, y_hat))