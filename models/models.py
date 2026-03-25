import math 
from typing import Dict, Tuple, Union

import torch 
import torch.nn as nn
from torchtyping import TensorType
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning as pl
from torch_geometric.data import Data

from models.utils import TransformerDecoder, GraphModule, PositionalEncoding, RMSELoss

        
class DecoderOnlyTransformer(pl.LightningModule):
    def __init__(self, general: Dict, optimizer: Dict, lr_scheduler: Dict, transformer: Dict, graph: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.graph = graph 
        
        self.criterion = RMSELoss()
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.transformer = transformer
        self.frame_embedding = general.frame_embedding 
        
        self.GraphEncoder = GraphModule(self.graph.encoder)
        self.GraphDecoder = GraphModule(self.graph.decoder)
        
        self.pe = PositionalEncoding(**self.transformer.positional_encoding)
        self.TransformerDecoder = TransformerDecoder(self.transformer)
        
        self.register_buffer(name="bos_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))
        self.register_buffer(name="sep_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))
        self.register_buffer(name="eos_graph", tensor=torch.rand(size=(1, self.graph.encoder.in_channels), dtype=torch.float32, device=self.device))  
        
        self.W_Q = torch.nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, self.transformer.encoder_layer.d_model, 1), nonlinearity="relu"))
        self.W_K = torch.nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, self.transformer.encoder_layer.d_model, 1), nonlinearity="relu"))
        self.W_V = torch.nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, 17, 1), nonlinearity="relu")) # TODO: Move max_n_players to config 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer)
        lr_scheduler = CosineAnnealingLR(optimizer, **self.lr_scheduler)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
            }
        
    def _loss(self, y: Data, y_hat: TensorType["*"], n_frames_target: int, n_players_target: int, mode: str) -> TensorType["*"]: 
        loss = self.criterion(y.x[-(n_frames_target+2)*n_players_target:-n_players_target, :2], y_hat[-(n_frames_target+1)*n_players_target:, :2]) # We are only interested in predicting the x- and y- coordintates 
        self.log_dict({'{}_loss'.format(mode): loss}, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def _attention(self, z: TensorType["n_frames", "hidden_dim"], t:int, p: int) -> TensorType["n_frames_target+2", "hidden_dim", "n_players_target"]:
        d = z.shape[-1]
        # we are only interested in the target part of the transformer output
        z = z[-t:, :].unsqueeze(1) # [t, 1, d] 

        # expansion of parameter matrices 
        W_Q = self.W_Q.repeat(t, 1, 1) # [t, d, 1]
        W_K = self.W_K.repeat(t, 1, 1) # [t, d, 1]
        
        W_V = self.W_V[:, :p, :].repeat(t, 1, 1) # [t, p, 1], p: number of output players 
        
        # matrix-vector multiplications (I do know that it is not actually a matrix, but a tensor...)
        Q = torch.bmm(W_Q, z) # [t, d, 1] @ [t, 1, d] -> [t, d, d]
        K = torch.bmm(W_K, z) # [t, d, 1] @ [t, 1, d] -> [t, d, d]
        V = torch.bmm(W_V, z).permute(0, 2, 1) # [t, p, 1] @ [t, 1, d] -> [t, p, d] -> [t, d, p]
        
        A = torch.bmm(torch.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(d), dim=-1), V) # [t, d, d] @ [t, d, p] -> [t, d, p]
        return A
    
    def _tensor2data(self, z: TensorType["n_frames_target+2", "hidden_dim", "n_players_target"]) -> Data:
        t, d, p = z.shape
        x = z.permute(0, 2, 1).reshape(-1, d)
        _edge = torch.nonzero((~torch.eye(p, dtype=torch.bool)).to(torch.int64).to(self.device)).T
        edge_index = torch.concat(tensors=[_edge+i for i in range(t)], dim=-1)
        
        data = Data(x=x, edge_index=edge_index)
        return data 
    
    def forward(self, seq: Data, seq_indices: TensorType["*"], n_frames_source: int, n_frames_target: int, n_players_source: int, n_players_target: int, iter: Union[None, int] = None) -> Tuple[TensorType["*"], TensorType["*"]]: 
        bos_graph = self.bos_graph.expand(n_players_source, -1)
        sep_graph = self.sep_graph.expand(n_players_target, -1)
        eos_graph = self.eos_graph.expand(n_players_target, -1)
        
        if iter is None or iter == 1: 
            seq.x[:n_players_source, :] = bos_graph # bos token is appended to the start of the sequence 
            seq.x[n_players_source*(n_frames_source+1):(n_players_source)*(n_frames_source+1)+n_players_target, :] = sep_graph # sep token is appended to the end of the input sequence 

        if iter is None: # only during training 
            seq.x[-n_players_target:, :] = eos_graph # eos token is not added to the sequence during prediction 
            
        seq_emb = self.GraphEncoder(seq.x, seq.edge_index, batch=seq_indices)
        seq_emb += self.pe._[:seq_emb.shape[0], :] 
        
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=seq_emb.shape[0], device=self.device, dtype=seq_emb.dtype)
        z = self.TransformerDecoder(seq_emb, mask) # [n_frames, d_model]
        z = self._attention(z=z, t=n_frames_target+2, p=n_players_target)
        data = self._tensor2data(z=z)  
           
        seq_node_hat = self.GraphDecoder(data.x, data.edge_index, batch=None)
        return seq, seq_node_hat
    
    def training_step(self, batch, batch_idx): 
        y, y_hat = self(**batch)
        loss = self._loss(y, y_hat, batch["n_frames_target"], batch["n_players_target"], "train")
        return loss 

    def validation_step(self, batch, batch_idx): 
        y, y_hat = self(**batch)
        loss = self._loss(y, y_hat, batch["n_frames_target"], batch["n_players_target"], "val")
        return loss 

    def test_step(self, batch, batch_idx): 
        y, y_hat = self(**batch)                            
        y, y_hat = self(**batch)
        loss = self._loss(y, y_hat, batch["n_frames_target"], batch["n_players_target"], "test")
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