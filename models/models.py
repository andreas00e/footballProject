import torch 
import torch.nn as nn 
import torch.nn.functional as F

import lightning as L 


class TransformerModel(L.LightningModule): 
    def __init__(self, feature_config: dict, transformer: dict, in_emb: dict, out_emb: dict):
        super().__init__()
        self.save_hyperparameters() 
    
        self.f_o_i = feature_config
        self.input_embedding = PlayerEmbeddingMLP(**in_emb)
        self.output_embedding = PlayerEmbeddingMLP(**out_emb)
        self.transformer = nn.Transformer(**transformer)
        self.linear = nn.Sequential( 
            nn.Linear(transformer['dim_feedforward'], 1000), 
            nn.ReLU(),
            nn.Linear(1000, 2)
        ) # TODO: adjust arcitecture to be more powerful/ sophisticated 

    def forward(self, src, tgt):
        src_emb, tgt_emb = self.input_embedding(src), self.output_embedding(tgt)
        return self.linear(self.transformer(src_emb, tgt_emb))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
     
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, x.shape[2], x.shape[3])
        y = y.view(-1, y.shape[2], y.shape[3])      
        y_hat = self(x, y)
        citerion = nn.MSELoss() # loss definition from: https://discuss.pytorch.org/t/rmse-loss-function/16540
        loss = torch.sqrt(citerion(y_hat, y)) # RMSE loss: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction
        return loss         

class PlayerEmbeddingMLP(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(), 
        ) # TODO: adjust arcitecture to be more powerful/ sophisticated 
        
    def forward(self, x): 
        return self.model(x)
    