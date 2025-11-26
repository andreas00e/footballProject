from typing import Tuple

import torch
from torchtyping import TensorType

max_in_players = 17 # XXX: If those numbers are bigger in one of the test files, we are f... (I mean, this just wouldn't work ...)
max_out_players = 8

def collate_fn(batch) -> Tuple[TensorType, TensorType]: 
    padded_features, padded_targets = [], []
    max_in_frames, max_out_frames = 0, 0
    
    for feature, target in batch: 
        if feature.shape[0] > max_in_frames: 
            max_in_frames = feature.shape[0]
        if target.shape[0] > max_out_frames: 
            max_out_frames = target.shape[0]
              
    for feature, target in batch: # element is of shape [frames, players, features (13 for input, 2 for output)] # pad players
        n_in_frames, n_in_players, _ = feature.shape
        
        frame_in_diff = max_in_frames - n_in_frames
        if frame_in_diff > 0: 
            feature_in_pad = torch.zeros((frame_in_diff, feature.shape[1], feature.shape[2]), 
                                         dtype=feature.dtype, device=feature.device)
            feature = torch.cat((feature, feature_in_pad), dim=0)
    
        
        player_in_diff = max_in_players - n_in_players
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
            
        player_out_diff = max_out_players - n_out_players
        if player_out_diff > 0: # ensure that there is something to be padded 
            player_out_pad = torch.zeros((target.shape[0], player_out_diff, target.shape[2]),
                                  dtype=target.dtype, device=target.device)
            target = torch.cat((target, player_out_pad), dim=1)
             
        padded_features.append(feature)
        padded_targets.append(target)
    
    features_batch = torch.stack(padded_features, dim=0)
    targets_batch = torch.stack(padded_targets, dim=0)
    
    return features_batch, targets_batch