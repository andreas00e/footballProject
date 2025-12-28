from typing import Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F 
from torchtyping import TensorType

@dataclass
class padding:         
    max_src_players: int = 17
    max_tgt_players: int = 17   
    # max_tgt_players: int = 9

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

        src = F.pad(src, (0, 0, 0, pad_src_players, 0, 0), "constant", -1) # src_frames, players, features
        tgt = F.pad(tgt, (0, 0, 0, pad_tgt_players, 0, 0), "constant", -1) # tgt_frames, players, features
        
        seq = torch.concat((src, tgt), dim=0) # src_frames + tgt_frames, players, features
        pad_seq_frames = max_frames - (src_frames + tgt_frames)
        seq = F.pad(seq, (0, 0, 0, 0, 0, pad_seq_frames), "constant", -1) # max_frames, players, features
        
        seq_list.append(seq)
        
    seq = torch.stack(tensors=seq_list, dim=1).permute(1, 0, 2, 3) # max_frames, batch, players, features 
    src_shapes = torch.tensor(src_shapes) 
    tgt_shapes = torch.tensor(tgt_shapes)
    
    return {
        "sequence": seq, 
        "src_shapes": src_shapes, 
        "tgt_shapes": tgt_shapes
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