from typing import Dict

import torch
import torch.nn.functional as F 
from torchtyping import TensorType

max_f_players = 17 # XXX: If those numbers are bigger in one of the test files, this wouldn't work ...
max_t_players = 9

def collate_fn(batch) -> Dict[str, TensorType['*']]: 
    max_frames = 0
    f_frames, t_frames = 0, 0 
    features, targets, features_shape, targets_shape = [], [], [], []
    
    for d in batch: 
        f_frames = d['features_shape'][0]
        t_frames = d['targets_shape'][0]

        if f_frames > max_frames: 
            max_frames = f_frames  
        if t_frames > max_frames: 
            max_frames = t_frames 
    
    for d in batch: # iterate over list elements len(batch) = batch_size 
        f_frames, f_players = d['features_shape'] # [frames, players, features]
        t_frames, t_players = d['targets_shape'] # [frames, players, features]  
        
        pad_f_players = max_f_players - f_players
        pad_t_players = max_t_players - t_players
        
        pad_f_frames = max_frames - f_frames 
        pad_t_frames = max_frames - t_frames 
        
        f = d['features']
        t = d['targets']
        
        f = F.pad(f, (0, 0, 0, pad_f_players, 0, pad_f_frames), "constant", 0)
        t = F.pad(t, (0, 0, 0, pad_t_players, 0, pad_t_frames), "constant", 0)

        features.append(f)
        targets.append(t)
        features_shape.append((f_frames, f_players))
        targets_shape.append((t_frames, t_players))
        
    features = torch.stack(features)
    targets = torch.stack(targets)
    features_shape = torch.tensor(features_shape, dtype=features.dtype, device=features.device)
    targets_shape = torch.tensor(targets_shape, dtype=targets.dtype, device=targets.device)
    
    data = {
        'features': features, 
        'targets': targets,
        'features_shape': features_shape,
        'targets_shape': targets_shape
    }
    
    return data 