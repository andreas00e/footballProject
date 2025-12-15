from typing import Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F 
from torchtyping import TensorType

@dataclass
class padding:         
    max_src_players: int = 17
    max_tgt_players: int = 9 
        

def collate_fn(batch) -> Dict[str, TensorType['*']]: 
    p = padding()
    srcs, tgts, = [], []
    
    src_max_frames, tgt_max_frames, src_frames, tgt_frames = 0, 0, 0, 0
    
    for element in batch: 
        src_frames = element["source_shape"][0]
        tgt_frames = element["target_shape"][0]

        if src_frames > src_max_frames: 
            src_max_frames = src_frames  
        if tgt_frames > tgt_max_frames: 
            tgt_max_frames = tgt_frames 
    
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