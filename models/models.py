from typing import Dict, List, Tuple, Union

import torch 
import torch.nn as nn 
from torchtyping import TensorType
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning as pl

from torch_geometric.data import Data

from models.utils import TransformerDecoder, GraphModule, PlayerEmbeddingMLP, PositionalEncoding, RMSELoss

        
class DecoderOnlyTransformer(pl.LightningModule):
    def __init__(self, general: Dict, optimizer: Dict, lr_scheduler: Dict, transformer: Dict, player_embedding: List, graph: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.graph = graph 
        
        self.criterion = RMSELoss()
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.frame_embedding = general.frame_embedding 
        
        self.SeqEmb = PlayerEmbeddingMLP(player_embedding) # Needed for non-graph frame embedding, e.g. sequential # TODO: Get rid of that for the moment, as we do not need it 
        self.GraphEncoder = GraphModule(self.graph.encoder)
        self.GraphDecoder = GraphModule(self.graph.decoder)
        
        self.pe = PositionalEncoding(**transformer.positional_encoding)
        self.TransformerDecoder = TransformerDecoder(transformer)
        
        self.register_buffer(name="bos_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))
        self.register_buffer(name="sep_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))
        self.register_buffer(name="eos_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))  
        
        self.temp = torch.nn.Parameter(data=torch.rand(size=(1,), device=self.device))
        
        self.min = torch.tensor([0.0], device=self.device)
        self.max = torch.tensor([0.0], device=self.device)

        
    def _loss(self, y: TensorType["*"], y_hat: TensorType["*"], mode: str) -> TensorType["*"]: 
        loss = self.criterion(y.x, y_hat) # only the x- and y-ćoordinates are considered for the loss 
        # loss = self.criterion(y.x[:self.eos_graph.shape[0], :], y_hat[self.bos_graph.shape[0]:, :]) # every feature is considered for the loss
        
        self.log_dict({'{}_loss'.format(mode): loss}, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss
    
    def _seq_2_nodes(self, seq: TensorType["*"], condition: TensorType["*"], n_frames_source: int, n_players_source: int, n_players_target: int) -> Tuple[Data, TensorType]:
        x_input = condition[:n_frames_source+2, :].repeat_interleave(repeats=n_players_source, dim=0)
        x_output = condition[n_frames_source+2:, :].repeat_interleave(repeats=n_players_target, dim=0)
        x = torch.concat(tensors=(x_input, x_output), dim=0)
        #  = torch.concat(tensors=(seq[:, :2], x), dim=-1)
        x = torch.clamp(input=(x+seq*self.temp), min=0.0, max=1.0)
        
        if not self.trainer.predicting: 
            self.log_dict({"temperature": self.temp}, prog_bar=False, batch_size=1)
        return x 
     
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer)
        lr_scheduler = CosineAnnealingLR(optimizer, **self.lr_scheduler)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
            }
    
    def forward(self, seq: Data, seq_indices: TensorType["*"], n_frames_source: int, n_frames_target: int, n_players_source: int, n_players_target: int, iter: Union[None, int] = None) -> Tuple[TensorType["*"], TensorType["*"]]: 
        bos_graph = self.bos_graph.expand(n_players_source, -1)
        sep_graph = self.sep_graph.expand(n_players_source, -1)
        eos_graph = self.eos_graph.expand(n_players_target, -1)
        
        if iter is None or iter == 1: 
            seq.x[:n_players_source, :] = bos_graph # bos token is appended to the start of the sequence 
            seq.x[n_players_source*(n_frames_source+1):(n_players_source)*(n_frames_source+2), :] = sep_graph # sep token is appended to the end of the input sequence 

        if iter is None: # only during training 
            seq.x[-n_players_target:, :] = eos_graph # eos token is not added to the sequence during prediction 
            
        seq_emb = self.GraphEncoder(seq.x, seq.edge_index, batch=seq_indices)
        seq_emb += self.pe._[:seq_emb.shape[0], :] 
        
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=seq_emb.shape[0], device=seq_emb.device, dtype=seq_emb.dtype)
        condition = self.TransformerDecoder(seq_emb, mask)
        
        if iter: 
            n_frames_target = iter
                   
        seq_hat_x = self._seq_2_nodes(seq.x, condition, n_frames_source, n_players_source, n_players_target)
                
        seq_node_hat = self.GraphDecoder(seq_hat_x, seq.edge_index, batch=None)
        return seq, seq_node_hat
    
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
    
    def predict_step(self, batch, batch_idx): # autoregressive prediction of the position of all target players for n_frames_targer, or when eos token is predicted 
        _, _, n_frames_source, n_frames_target, n_players_source, n_players_target = batch.values()
        print(f"Given {n_players_source} input players for {n_frames_source} frames, \n \
              we are predicting the position of {n_players_target} players for {n_frames_target} frames.")
        
        for i in range(n_frames_target): 
            if i == 0: 
                y, y_hat = self(**batch, iter=i+1)
            else: 
                pass 
            
        print(f"<bos> token: {self.bos_graph}")   
        print(f"<sep> token: {self.sep_graph}") 
        print(f"<eos> token: {self.eos_graph}")     
        
        return y.x[:, :2], y_hat[:, :2]