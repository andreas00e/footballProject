from typing import Dict, List, Tuple

import torch 
import torch.nn as nn 
from torchtyping import TensorType
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning as pl

from torch_geometric.data import Data

from models.utils import GraphModule, PlayerEmbeddingMLP, PositionalEncoding, RMSELoss

        
class DecoderOnlyTransformer(pl.LightningModule):
    def __init__(self, general: Dict, optimizer: Dict, lr_scheduler: Dict, transformer: Dict, player_embedding: List, graph: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.graph = graph 
        
        self.criterion = RMSELoss()
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.frame_embedding = general.frame_embedding 
        
        self.SeqEmb = PlayerEmbeddingMLP(player_embedding)
        self.GraphEncoder = GraphModule(self.graph.encoder)
        self.GraphDecoder = GraphModule(self.graph.decoder)
        
        self.pe = PositionalEncoding(**transformer.positional_encoding)
        self.encoder_layer = nn.TransformerEncoderLayer(**transformer.encoder_layer)
        self.TransformerDecoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer.decoder.num_layers)
        
        self.register_buffer(name="bos_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))
        self.register_buffer(name="sep_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))
        self.register_buffer(name="eos_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))

    def _loss(self, y: TensorType["*"], y_hat: TensorType["*"], mode: str) -> TensorType["*"]: 
        loss = self.criterion(y, y_hat)
        self.log_dict({'{}_loss'.format(mode): loss}, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss
    
    def _seq_2_nodes(self, n_frames: int, n_players: int, condition: TensorType) -> Tuple[Data, TensorType]:
        x = condition[-1-n_frames:, :].repeat_interleave(repeats=n_players, dim=0) # frame-based information for every output player  
        edge_index = torch.cat(tensors=[torch.nonzero(~torch.eye(n_players, dtype=torch.bool, device=self.device))+i*n_players for i in range(n_frames)], dim=0).T
        return Data(x=x, edge_index=edge_index)
     
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer)
        lr_scheduler = CosineAnnealingLR(optimizer, **self.lr_scheduler)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
            }
    
    def forward(self, seq, seq_indices, n_frames_source, n_frames_target, n_players_source, n_players_target): 
        bos_graph = self.bos_graph.expand(n_players_source, -1)
        sep_graph = self.sep_graph.expand(n_players_source, -1)
        eos_graph = self.eos_graph.expand(n_players_target, -1)
        
        seq.x[:n_players_source, :] = bos_graph
        seq.x[n_players_source*(n_frames_source+1):(n_players_source)*(n_frames_source+2), :] = sep_graph
        seq.x[-n_players_target:, :] = eos_graph
            
        seq_emb = self.GraphEncoder(seq.x, seq.edge_index, batch=seq_indices)
        seq_emb += self.pe.pe[:seq_emb.shape[0], :].squeeze() # TODO: Change to self.pe
        
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=seq_emb.shape[0], device=seq_emb.device, dtype=seq_emb.dtype)
        condition = self.TransformerDecoder(seq_emb, mask)
    
        seq_hat = self._seq_2_nodes(n_frames_target, n_players_target, condition)
        seq_node_hat = self.GraphDecoder(seq_hat.x, seq_hat.edge_index, batch=None)
        return seq.x[-seq_node_hat.shape[0]:, :2], seq_node_hat[:, :2] # The only parts of the prediction we are interested in are the predicted x- and y-coordinates
    
    def training_step(self, batch, batch_idx): 
        y, y_hat = self(**batch)                              
        loss = self._loss(y, y_hat, "train")
        return loss 

    def validation_step(self, batch, batch_idx): 
        y, y_hat = self(**batch)                              
        loss = self._loss(y, y_hat, "val")
        return loss 

    def test_step(self, batch, batch_idx): 
        y, y_hat = self(**batch)                            
        loss = self._loss(y, y_hat, "test")
        return loss 
    
    def predict_step(self, batch, batch_idx):
        return self(**batch)