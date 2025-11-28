import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchtyping import TensorType

import lightning as L 
from scipy.optimize import linear_sum_assignment


class TransformerModel(L.LightningModule): 
    def __init__(self, feature_config: dict, size_window: int, transformer: dict, in_emb: dict, out_emb: dict):
        super().__init__()
        self.save_hyperparameters() 
    
        self.f_o_i = feature_config
        self.size_window = size_window
        self.f_embedding = PlayerEmbeddingMLP(**in_emb)
        self.t_embedding = PlayerEmbeddingMLP(**out_emb)

        self.transformer = nn.Transformer(**transformer)
        self.linear = nn.Sequential( 
            nn.Linear(transformer['dim_feedforward'], 1000), 
            nn.ReLU(),
            nn.Linear(1000, 2) # for every player and every frame? 
        ) # TODO: adjust arcitecture to be more powerful/ sophisticated
        
    def forward(self, f, t):
        f_emb, t_emb = self.f_embedding(f), self.t_embedding(t) # [batch*frames, players, embedded_features]
        return self.linear(self.transformer(f_emb, t_emb))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
    def training_step(self, batch, batch_idx): 
        f, t = batch['features'], batch['targets'] # [batch, frames, players, features]
        f_shape, t_shape = batch['features_shape'], batch['targets_shape']
        
        f = f.view(-1, f.shape[2], f.shape[3]) # [batch*frames, players, features]
        t = t.view(-1, t.shape[2], t.shape[3])
        
        criterion = nn.MSELoss()
        y_hat = self(f, t)
        loss = torch.sqrt(criterion(y_hat, t))
        
        self.log_dict({'train_loss' : loss})
    
    def validation_step(self, batch, batch_idx): 
        f, t = batch['features'], batch['targets'] # [batch, frames, players, features]
        f_shape, t_shape = batch['features_shape'], batch['targets_shape']
        
        f = f.view(-1, f.shape[2], f.shape[3]) # [batch*frames, players, features]
        t = t.view(-1, t.shape[2], t.shape[3])
        
        criterion = nn.MSELoss()
        y_hat = self(f, t)
        loss = torch.sqrt(criterion(y_hat, t))
        
        self.log_dict({'val_loss' : loss})
    
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
    