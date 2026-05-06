from typing import Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F 
from torchtyping import TensorType

from torch_geometric.data import Data


@dataclass
class padding:         
    max_src_players: int = 17
    max_tgt_players: int = 17   
    # max_tgt_players: int = 9

def collate_fn_graph(batch): # TODO: Move (some of) this to dataloading    
    for b in batch: # XXX: Currently this only works for batch_size=1
        source, target, n_frames_source, n_frames_target, n_players_source, n_players_target, n_features, out_ids = b.values()
        out_ids = torch.tensor(out_ids)


        placeholder_token_source = torch.zeros(size=(n_players_source, n_features), dtype=torch.float32) 
        placeholder_token_target = torch.zeros(size=(n_players_target, n_features), dtype=torch.float32) 
        
        if target: # mode=train
            x = torch.concat(tensors=(placeholder_token_source, source.x, placeholder_token_target, target.x, placeholder_token_target), dim=0) # feature vector for training 
        else: # mode=predict
            x = torch.concat(tensors=(placeholder_token_source, source.x, placeholder_token_target), dim=0) # feature vector for inference 
        
        bos_edge_index = torch.nonzero((~torch.eye(n_players_source, dtype=torch.bool)).to(torch.int64)).T # adjacency vector for the nodes constituting the <bos> token 
        eos_edge_index = torch.nonzero((~torch.eye(n_players_target, dtype=torch.bool)).to(torch.int64)).T # adjacency vector for the nodes constituting the <sep>, and <eos> token 
        
        source_edge_index = source.edge_index + n_players_source 
        
        if n_players_target > 1: 
            sep_edge_index = (eos_edge_index + torch.max(source_edge_index) + 1) 
        elif n_players_target == 1: 
            sep_edge_index = torch.empty((2, 0), dtype=torch.int64) # <sep> token consists of one node, and consequently no edges
        else: 
            raise ValueError("The number of output players has to be positive and bigger than zero!")
            
        if target: # mode=train
            if sep_edge_index.numel() != 0 and eos_edge_index.numel() != 0: # if the <sep> token is empty, so is the <eos> token 
                eos_edge_index += torch.max(sep_edge_index)     
            edge_index = torch.concat(tensors=(bos_edge_index, source.edge_index, sep_edge_index, target.edge_index, eos_edge_index), dim=-1) # mode=train
            target_indices = [i for i in range(n_frames_target+2) for _ in range(n_players_target)] # include <sep>, and <eos> token

        else: # mode=predict
            edge_index = torch.concat(tensors=(bos_edge_index, source.edge_index, sep_edge_index), dim=-1) # mode=predict
            target_indices = [0]*n_players_target # <sep> token after the input sequence and <bos>
        
        seq = Data(x=x[:, :-1], edge_index=edge_index) # number of predicted output frames is currently not passed to model in any way
        source_indices = [i for i in range(n_frames_source+1) for _ in range(n_players_source)] # indices for input sequence including the <bos> token 
    
    seq_indices = source_indices + [i+max(source_indices)+1 for i in target_indices] 
    seq_indices = torch.tensor(seq_indices, dtype=torch.int64)
    
    return {
        "seq": seq, 
        "seq_indices": seq_indices, 
        "n_frames_source": n_frames_source,
        "n_frames_target": n_frames_target, 
        "n_players_source": n_players_source, 
        "n_players_target": n_players_target, 
        "out_ids": out_ids
        }

def collate_fn(batch) -> Dict[str, TensorType['*']]: 
    p = padding()
    seq_list, src_shapes, tgt_shapes = [], [], [] 
    max_frames = 0 
    
    for element in batch: 
        src_frames, src_players = element["source_shape"]
        tgt_frames, tgt_players = element["target_shape"]      
        
        src_shapes.append([src_frames, src_players])   
        tgt_shapes.append([tgt_frames, tgt_players])
        
        frames = src_frames + tgt_frames 
        if frames > max_frames: 
            max_frames = frames 
    
    for i, element in enumerate(batch): 
        src = element["source"] 
        tgt = element["target"] 
        
        src_frames, src_players = src_shapes[i]
        tgt_frames, tgt_players = tgt_shapes[i]
        
        pad_src_players = p.max_src_players - src_players # number of padding entries for the source players 
        pad_tgt_players = p.max_tgt_players - tgt_players # number of padding entries for the target players 

        src = F.pad(src, (0, 0, 0, pad_src_players, 0, 0), "constant", -1) # [src_frames, players, features]
        tgt = F.pad(tgt, (0, 0, 0, pad_tgt_players, 0, 0), "constant", -1) # [tgt_frames, players, features]
        tkn = torch.ones(size=([1]+[*src.shape[1:]])) * -2 # [1, players, features]

        seq = torch.concat(tensors=(tkn, src, tkn, tgt, tkn), dim=0) # [1+src_frames+1+tgt_frames+1, players, features]
        pad_seq_frames = max_frames - (src_frames + tgt_frames)
        seq = F.pad(seq, (0, 0, 0, 0, 0, pad_seq_frames), "constant", -1) # [max_frames+3, players, features]
        seq_list.append(seq)
        
    seq = torch.stack(tensors=seq_list, dim=1) # max_frames, batch, players, features 
    src_s = torch.tensor([s[0] for s in src_shapes]) # number of source frames for each element in the sequence 
    tgt_s = torch.tensor([t[0] for t in tgt_shapes]) # number of target frames for each element in the sequence
    
    return {
        "seq": seq, 
        "src_shapes": src_s, 
        "tgt_shapes": tgt_s
    }

def collate_fn_gap(batch) -> Dict[str, TensorType['*']]: 
    p = padding()
    srcs, tgts, = [], []
    src_max_frames, tgt_max_frames, src_frames, tgt_frames = [0,]*4
    
    for element in batch: 
        src_frames = element["source_shape"][0]
        tgt_frames = element["target_shape"][0]

        if src_frames > src_max_frames: 
            src_max_frames = src_frames # max number of frames in the input sequence
        if tgt_frames > tgt_max_frames: 
            tgt_max_frames = tgt_frames # max number of frames in the output sequence
    
    for element in batch: 
        src_frames, src_players = element["source_shape"] # src_frames, src_players, src_features
        tgt_frames, tgt_players = element["target_shape"] # tgt_frames, tgt_players, tgt_features  
        
        pad_src_players = p.max_src_players - src_players
        pad_tgt_players = p.max_tgt_players - tgt_players
        
        pad_s_frames = src_max_frames - src_frames 
        pad_t_frames = tgt_max_frames - tgt_frames 
        
        src = element["source"]
        tgt = element["target"]
            
        src = F.pad(src, (0, 0, 0, pad_src_players, 0, pad_s_frames), "constant", -1)
        tgt = F.pad(tgt, (0, 0, 0, pad_tgt_players, 0, pad_t_frames), "constant", -1)

        srcs.append(src), tgts.append(tgt)
        
    srcs = torch.stack(srcs).permute(1, 0, 2, 3) # src_frames, batch, src_players, src_features 
    tgts = torch.stack(tgts).permute(1, 0, 2, 3) # tgt_frames, batch, tgt_players, tgt_features 
    
    return {
        "source": srcs, 
        "target": tgts,
    }