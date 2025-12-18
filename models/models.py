from typing import Dict, Tuple, Union
import torch 
import torch.nn as nn 
from torchtyping import TensorType
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning as pl

class DecoderOnlyTransformer(pl.LightningModule):
    def __init__(self, model_conf: dict, in_emb: dict):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_conf: Dict[str: Union[int, None]] = model_conf 
        self.seq_token_shape = [1, 1, self.model_conf.players, self.model_conf.features]
        self.batch_size = self.model_conf.batch
        self.io_last_dim = int(self.model_conf.i.output_dim * self.model_conf.token_factor)
        
        in_emb["input_dim"] += self.io_last_dim # TODO: Move to conf

        self.criterion = nn.MSELoss() # TODO: Change self.criterion to RMSE
        
        self.seq_emb = PlayerEmbeddingMLP(**in_emb)
        self.encoder_layer = nn.TransformerEncoderLayer(**self.model_conf.transformer) # TODO: change d_model and dim_feedforward 
        self.decoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.pos = PosMLP(self.model_conf)   
        
    def on_fit_start(self):
        self.register_buffer(name="bos", tensor=torch.rand(size=self.seq_token_shape, dtype=torch.float32, device=self.device)) # begin of sequence token ([0, 1))
        self.bos = self.bos.expand([-1, self.batch_size, -1, -1]) # [1, batch, players, features] 
        self.register_buffer(name="sep", tensor=torch.rand(size=self.seq_token_shape, dtype=torch.float32, device=self.device)) # seperator token ([0, 1))
        self.sep = self.sep.expand([-1, self.batch_size, -1, -1]) # [1, batch, players, features] 
        self.register_buffer(name="eos", tensor=torch.rand(size=self.seq_token_shape, dtype=torch.float32, device=self.device)) # end of sequence token ([0, 1))
        self.eos = self.eos.expand([-1, self.batch_size, -1, -1]) # [1, batch, players, features] 

        self.register_buffer(name="ins", tensor=torch.rand(size=[1,]*3+[self.io_last_dim], dtype=torch.float32, device=self.device)) # input identifier token   
        self.ins = self.ins.expand([-1, self.batch_size, self.model_conf.players, -1]) # [1, batch, players, features+self.io_last_dim] 
        self.register_buffer(name="outs", tensor=torch.rand(size=[1]*3+[self.io_last_dim], dtype=torch.float32, device=self.device)) # output identifier token    
        self.outs = self.outs.expand([-1, self.batch_size, self.model_conf.players, -1]) # [1, batch, players, features+self.io_last_dim] 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_conf.optimizer.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.model_conf.lr_scheduler.T_max, eta_min=self.model_conf.lr_scheduler.eta_min)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
            }

    def _loss(self, y, y_hat, mode: str): # y: [1+src_frames+1+tgt_frames, ...], y_hat:  # [src_frames+1+tgt_frames+1, ...]
        loss = self.criterion(y[:-1, ...], y_hat[1:, ...])        
        self.log_dict({'{}_loss'.format(mode): loss}, batch_size=y.shape[1], prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def _prepare_tokens(self, src: TensorType["1", "batch_size", "1", "1"], tgt: TensorType["1", "batch_size", "1", "1"]) \
                        -> Tuple[TensorType["1+srcframes+1", "batch_size", "1", "1"], TensorType["tgt_frames+1", "batch_size", "1", "1"]]:
        """
        Method adjusting the shape of the tokens marking the input as input, and the output as output

        Returns:
            Tuple[TensorType["*"], TensorType["*"]: The adjusted input and output tokens
        """
        ins = self.ins.expand([src.shape[0]+2]+[-1]*3) # [1+src_frames+1, batch, players, emb_features] 
        outs = self.outs.expand([tgt.shape[0]+1]+[-1]*3) # [tgt_frames+1, batch, players, emb_features] 
        return ins, outs

    def _prepare_sequence(self, batch: TensorType["*"]) -> TensorType["*"]: 
        """ 
        Method concatenating the input sequence, the output sequence, and the sequence tokens ("bos", "sep", "eos")

        Returns:
            TensorType['*']: Concatenated sequence consisting of the input sequence, the output sequence, and the sequence tokens ("bos", "sep", "eos")
        """
        src = batch["source"] # [src_frames, batch, players, features]
        tgt =  batch["target"] # [tgt_frames, batch, players, features]
        
        ins, outs = self._prepare_tokens(src=src, tgt=tgt)
        
        seq = torch.concat(tensors=(self.bos, src, self.sep, tgt, self.eos), dim=0) # [1+src_frames+1+tgt_frames+1, batch, players, features]
        return seq, ins, outs 
    
    def forward(self, seq: TensorType, ins: TensorType, outs: TensorType): 
        seq_mask = (seq == -1).all(-1) # [1+src_frames+1+tgt_frames+1, batch, players] # XXX: all() or any()? 
        seq = torch.concat(tensors=(seq, torch.concat(tensors=(ins, outs), dim=0)), dim=-1) # [1+src_frames+1+tgt_frames+1, batch, players, features+io_last_dim]
        
        seq_emb = self.seq_emb(seq) # [1+src_frames+1+tgt_frames+1, batch, players, emb_features]
        seq_emb *= ~seq_mask.unsqueeze(-1).expand([-1,]*3+[seq_emb.shape[-1]])
        seq_emb = seq_emb.sum(-2).squeeze() # [1+src_frames+1+tgt_frames+1, batch, emb_features] # TODO: Implement more sophisticated nn.Module
        
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=seq_emb.shape[0], device=self.device, dtype=seq_emb.dtype)
        src_key_padding_mask = seq_mask.all(-1).permute(1, 0).to(torch.float32) # [batch, 1+src_frames+1+tgt_frames+1]
        
        seq_hat = self.decoder(src=seq_emb, mask=mask, src_key_padding_mask=src_key_padding_mask) # [1+src_frames+1+tgt_frames+1, batch, emb_features]
        seq_hat = seq_hat.unsqueeze(-2).expand([-1]*2+[seq.shape[-2]]+[-1]) # [1+src_frames+1+tgt_frames+1, batch, players, emb_features]
        seq_hat = self.pos(seq_hat) # [1+src_frames+1+tgt_frames+1, batch, players, features]
        seq_hat *= ~seq_mask.unsqueeze(-1).expand([-1,]*3+[seq_hat.shape[-1]]) # mask out frame and player paddings 
        return seq_hat
    
    def training_step(self, batch, batch_idx): 
        y, ins, outs = self._prepare_sequence(batch) # [1+src_frames+1+tgt_frames+1, batch, players, features]
        y_hat = self(y, ins, outs) # [1+src_frames+1+tgt_frames+1, batch, players, features]                                                    
        loss = self._loss(y, y_hat, "train")
        return loss 

    def validation_step(self, batch, batch_idx): 
        y, ins, outs = self._prepare_sequence(batch) # [1+src_frames+1+tgt_frames+1, batch, players, features]
        y_hat = self(y, ins, outs) # [1+src_frames+1+tgt_frames+1, batch, players, features]                                                       
        loss = self._loss(y, y_hat, "val")
        return loss 

    def test_step(self, batch, batch_idx): 
        y, ins, outs = self._prepare_sequence(batch) # [1+src_frames+1+tgt_frames+1, batch, players, features]
        y_hat = self(y, ins, outs) # [1+src_frames+1+tgt_frames+1, batch, players, features]                                                         
        loss = self._loss(y, y_hat, "test")
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
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.output_dim), 
        )
     
    def forward(self, x): 
        return self.model(x)

class PosMLP(nn.Module): 
    def __init__(self, model_conf: dict):
        super().__init__()
        
        self.feature = nn.Sequential( 
            nn.Linear(model_conf.i.output_dim, model_conf.pos.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_conf.pos.hidden_dim, model_conf.pos.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(model_conf.pos.hidden_dim, model_conf.pos.output_dim),
        )
    
    def forward(self, x): 
        return self.feature(x)