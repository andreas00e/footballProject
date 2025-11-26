from typing import Tuple

import torch
from torchtyping import TensorType

MAX_IN_PLAYERS = 13 # XXX: If those numbers are bigger in one of the test files, we are f... (I mean, this just wouldn't work ...)
MAX_OUT_PLAYERS = 5

def collate_fn(batch) -> Tuple[TensorType, TensorType]: 
    out = []
    
    for feature, target in batch: # element is of shape [frames, players, features (13 for input, 2 for output)] # pad players 
        n_in_players = feature.shape[1] 
        in_diff = MAX_IN_PLAYERS - n_in_players
        if in_diff > 0: # ensure that there is something to be padded 
            in_pad = torch.zeros((feature.shape[0], in_diff, feature.shape[2]))
            feature = torch.cat((feature, in_pad), dim=1)

        n_out_players = target.shape[1]
        out_diff = MAX_OUT_PLAYERS - n_out_players
        if out_diff > 0: # ensure that there is something to be padded 
            out_pad = torch.zeros((target.shape[0], out_diff, target.shape[2]))
            target = torch.cat((target, out_pad), dim=1)
        
        out.append((feature, target))
    
    return out