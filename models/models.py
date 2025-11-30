import numpy as np 
from typing import Dict 

import torch 
import torch.nn as nn 
import lightning as L 

from data.utils import hungarian_matching

class TransformerModel(L.LightningModule): 
    def __init__(self, feature_config: dict, size_window: int, transformer: dict, in_emb: dict, out_emb: dict):
        super().__init__()
        self.save_hyperparameters() 
        self.f_o_i = feature_config
        self.size_window = size_window
        
        self.s_embedding = SmallPlayerEmbeddingMLP(**in_emb)
        self.t_embedding = SmallPlayerEmbeddingMLP(**out_emb)
        self.s_frame_embedding = FrameEmbeddingTF(transformer)
        self.t_frame_embedding = FrameEmbeddingTF(transformer)

        self.transformer = nn.Transformer(**transformer)
                
        self.pos = nn.Sequential( 
            nn.Linear(transformer['dim_feedforward'], 1000), 
            nn.ReLU(),
            nn.Linear(1000, 2) 
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
    def loss(self, t, t_hat, mode: str): 
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(t, t_hat))        
        self.log_dict({'{}_loss'.format(mode): loss})
        return loss
    
    def player_mask(self, shapes_o, shapes_p, device):
        shapes_o = shapes_o.cpu().numpy().astype(np.int32)
        mask = []
        for s in shapes_o: 
            ones = torch.ones(1, shapes_p[1], s , 256, device=device)
            zeros = torch.zeros(1, shapes_p[1], shapes_p[2]-s, 256, device=device)
            s_mask = torch.concat((ones, zeros), dim=2)
            mask.append(s_mask)
            
        mask = torch.concat(mask, dim=0)
        return mask.view(-1, mask.shape[-2], mask.shape[-1])
        
    def forward(self, s, t, s_player_mask, t_player_mask):
        s = s.view(-1, s.shape[2], s.shape[3]) # [batch*frames, s_players, s_features]
        t = t.view(-1, t.shape[2], t.shape[3]) # [batch*frames, t_players, t_features]
        
        s_emb = self.s_embedding(s) # [batch*frames, s_players, emb_features]
        t_emb = self.t_embedding(t) # [batch*frames, t_players, emb_features]
        
        s_frame_emb = self.s_frame_embedding(s_emb, s_player_mask) # [batch*frames, s_players, emb_features]
        t_frame_emb = self.t_frame_embedding(t_emb, t_player_mask) # [batch*frames, t_players, emb_features]
        
        t_hat = self.pos(self.transformer(s_frame_emb, t_frame_emb)) # [batch*frames, players, 2]
        return t_hat, t
            
    def training_step(self, batch, batch_idx): 
        s, t = batch['sources'], batch['targets'] # [batch, frames, s_/t_players, s_/t_features]
        s_shape, t_shape = batch['sources_shape'], batch['targets_shape']
        
        s_player_mask = self.player_mask(shapes_o=s_shape[:, 1], shapes_p=s.shape, device=s.device)
        t_player_mask = self.player_mask(shapes_o=t_shape[:, 1], shapes_p=t.shape, device=t.device)
        
        t_hat, t = self(s, t, s_player_mask, t_player_mask)
        loss = self.loss(t, t_hat, 'train')
        return loss 
    
    def validation_step(self, batch, batch_idx): 
        s, t = batch['sources'], batch['targets'] # [batch, frames, s_/t_players, s_/t_features]
        s_shape, t_shape = batch['sources_shape'], batch['targets_shape']
        
        s_player_mask = self.player_mask(shapes_o=s_shape[:, 1], shapes_p=s.shape, device=s.device)
        t_player_mask = self.player_mask(shapes_o=t_shape[:, 1], shapes_p=t.shape, device=t.device)
        
        t_hat, t = self(s, t, s_player_mask, t_player_mask)
        _ = self.loss(t, t_hat, 'val')
    
class PlayerEmbeddingMLP(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim), 
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(), 
        )
        
    def forward(self, x): 
        return self.model(x)
    
class SmallPlayerEmbeddingMLP(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        
        if hidden_dim: 
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim), 
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim), 
                nn.ReLU()
            ) 
        else: 
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim), 
                nn.ReLU()
            )
     
    def forward(self, x): 
        return self.model(x)
    
class FrameEmbeddingTF(nn.Module):
    def __init__(self, transformer: Dict[str, int]):
        super().__init__() 
        self.frame_encoder_layer = nn.TransformerEncoderLayer(**transformer)
        self.frame_encoder = nn.TransformerEncoder(self.frame_encoder_layer, num_layers=1)
                       
    def forward(self, x, mask): 
        x = torch.mul(x, mask)
        frame = self.frame_encoder(x)
        return frame      