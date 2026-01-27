from typing import Dict, Iterable, List, Tuple, Union

import torch 
import torch.nn as nn 
from torchtyping import TensorType
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning as pl

from torch_geometric.data import Data
from torch_geometric.nn.models import GAT 
from torch_geometric.nn.pool import global_mean_pool

from models.utils import PositionalEncoding

        
class DecoderOnlyTransformer(pl.LightningModule):
    def __init__(self, general: Dict, optimizer: Dict, lr_scheduler: Dict, transformer: Dict, player_embedding: Dict, graph: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = RMSELoss()
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.frame_embedding = general.frame_embedding 
        
        self.seq_emb = PlayerEmbeddingMLP(player_embedding)
        self.graph_encoder = GraphModule(graph.encoder)
        self.graph_decoder = GraphModule(graph.decoder)
        
        self.encoder_layer = nn.TransformerEncoderLayer(**transformer.encoder_layer)
        self.transformer_decoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer.decoder)
        self.pe = PositionalEncoding(transformer.positional_encoding)
    
        self.bos_graph, self.sep_graph = [torch.rand(size=(1, graph.encoder.in_channels))]*2
        self.eos_graph = torch.rand(size=(1, graph.encoder.in_channels))

    def _loss(self, y: TensorType["*"], y_hat: TensorType["*"], mode: str) -> TensorType["*"]: # XXX: Ensure that correct tensors are passed to the loss function 
        loss = self.criterion(y, y_hat)
        self.log_dict({'{}_loss'.format(mode): loss}, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def _reconstruct_nodes(self, n_frames: int, n_players: int, condition: TensorType) -> Tuple[Data, TensorType]:
        x = condition[-1-n_frames:, :].unsqueeze(0).repeat(n_players, 1, 1).view(n_players*(n_frames+1), -1)
        edge_index = torch.cat(tensors=[torch.nonzero(~torch.eye(n_players, dtype=torch.bool, device=self.device))+i*n_players for i in range(n_frames)], dim=0).T
        batch = torch.tensor([i+j for i in range(n_players) for j in range(n_frames)], dtype=torch.int64, device=self.device)
        return Data(x=x, edge_index=edge_index), batch
     
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer)
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
        
            
        seq_emb = self.graph_encoder(seq.x, seq.edge_index, batch=seq_indices)
        seq_emb += self.pe.pe[:seq_emb.shape[0], 0, :].squeeze()
        
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=seq_emb.shape[0], device=seq_emb.device, dtype=seq_emb.dtype)
        condition = self.transformer_decoder(seq_emb, mask)
    
        seq_hat, _ = self.reconstruct_nodes(n_frames_target, n_players_target, condition) # I also need to know how many nodes should be put out
        seq_node_hat = self.graph_decoder(seq_hat.x, seq_hat.edge_index, batch=False)
        return seq.x, seq_node_hat
    
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

    # def predict_step(self, batch, batch_idx):
    #     index = 0
    #     seq = self(batch, index)
    #     while torch.dist(seq[-1:, :], self.eos_graph) < 0.0001 or index <= 30: 
    #         index += 1
    #         seq = torch.concat(tensors=(seq, self(seq, index)[-1:, :]), dim=0)

    #     return seq 
    
 
def list2sequential(features: Iterable) -> List: 
    return [layer for i in range(len(features)-1) for layer in (nn.Linear(features[i], features[i+1]), nn.ReLU())][:-1]

class RMSELoss(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.mse = nn.MSELoss() 
        
    def forward(self, y, y_hat): 
        return torch.sqrt(self.mse(y, y_hat))
           
class PlayerEmbeddingMLP(nn.Module): 
    def __init__(self, player_embedding: dict):
        super().__init__()
        
        self.model = nn.Sequential(list2sequential(player_embedding.values()))
        
    def forward(self, x): 
        return self.model(x)
    
class GraphModule(nn.Module): 
    def __init__(self, args: dict) -> None:
        super().__init__()
        
        self.model = GAT(**args)
    
    def forward(self, x: TensorType["n_nodes", "n_features"], edge_index: TensorType["2", "n_edges"], 
                batch: Union[bool | TensorType["1", "num_nodes"]]) -> TensorType["*"]:
        
        x = self.model(x, edge_index)
        if batch:
            x = global_mean_pool(x, batch)
        return x