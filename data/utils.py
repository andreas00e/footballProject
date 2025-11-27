from typing import Dict, Tuple

import torch
from torchtyping import TensorType

import torch.nn.functional as F 

max_f_players = 17 # XXX: If those numbers are bigger in one of the test files, this wouldn't work ...
max_t_players = 9

def collate_fn(batch) -> Tuple[TensorType, TensorType]: 
    padded_features, padded_targets = [], []
    max_in_frames, max_out_frames = 0, 0
    
    for feature, target in batch: 
        if feature.shape[0] > max_in_frames: 
            max_in_frames = feature.shape[0]
        if target.shape[0] > max_out_frames: 
            max_out_frames = target.shape[0]
    
    if max_out_frames >= max_in_frames: 
        max_in_frames = max_out_frames
    else: 
        max_out_frames = max_in_frames
              
    for feature, target in batch: # element is of shape [frames, players, features (13 for input, 2 for output)] # pad players
        n_in_frames, n_in_players, _ = feature.shape
        
        frame_in_diff = max_in_frames - n_in_frames
        if frame_in_diff > 0: 
            feature_in_pad = torch.zeros((frame_in_diff, feature.shape[1], feature.shape[2]), 
                                         dtype=feature.dtype, device=feature.device)
            feature = torch.cat((feature, feature_in_pad), dim=0)
    
        
        player_in_diff = max_f_players - n_in_players
        if player_in_diff > 0: # ensure that there is something to be padded 
            player_in_pad = torch.zeros((feature.shape[0], player_in_diff, feature.shape[2]), 
                                 dtype=feature.dtype, device=feature.device)
            feature = torch.cat((feature, player_in_pad), dim=1)
                
        n_out_frames, n_out_players, _ = target.shape
        
        frame_out_diff = max_out_frames - n_out_frames
        if frame_out_diff > 0: 
            frame_out_pad = torch.zeros((frame_out_diff, target.shape[1], target.shape[2]), 
                                         dtype=target.dtype, device=target.device)
            target = torch.cat((target, frame_out_pad), dim=0)
            
        player_out_diff = max_t_players - n_out_players
        if player_out_diff > 0: # ensure that there is something to be padded 
            player_out_pad = torch.zeros((target.shape[0], player_out_diff, target.shape[2]),
                                  dtype=target.dtype, device=target.device)
            target = torch.cat((target, player_out_pad), dim=1)
             
        padded_features.append(feature)
        padded_targets.append(target)
    
    features_batch = torch.stack(padded_features, dim=0)
    targets_batch = torch.stack(padded_targets, dim=0)
    
    return features_batch, targets_batch

def collate_fn_dict(batch) -> Dict[str, TensorType['*']]: 
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