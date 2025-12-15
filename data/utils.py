from typing import Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F 
from torchtyping import TensorType


@dataclass
class padding:         
    max_source_players: int = 17 # XXX: If those numbers are bigger in one of the test files, this wouldn't work ...
    max_target_players: int = 9 

def collate_fn(batch): # XXX: batch-wise frame padding, global player padding
    p = padding()
    max_frames = 0
    sources, targets = [], []

    for element in batch: # XXX: batch-wise padding 
        source_frames = element["source_shape"][0] 
        target_frames = element["target_shape"][0]
        
        if source_frames + target_frames > max_frames: 
            max_frames = source_frames + target_frames 

    for element in batch: 
        source = element["source"] # frames, players, features
        target = element["target"] # frames, players, features 
        source_frames, source_players, source_features = element["source_shape"]
        target_frames, target_players, target_features = element["target_shape"]
        
        frame_pad = max_frames - (source_frames + target_frames)
        source_players_pad = p.max_source_players - source_players
        target_players_pad = p.max_target_players - target_players
        target_features_pad = source_features - target_features

        source = F.pad(input=source, pad=[0, 0, 0, source_players_pad, 0, 0], value=-2)          
        target = F.pad(input=target, pad=[0, target_features_pad, 0, target_players_pad, 0, frame_pad], mode="constant", value=-2) # pad: feature_left, feature_right, player_left, ..., frame_right
        
        sources.append(source)
        targets.append(targets)

        
    return {
        "source": torch.stack(sources, dim=0),
        "target": target,
    }
        

# def collate_fn(batch) -> Dict[str, TensorType['*']]: 
#     max_frames = 0
#     s_frames, t_frames = 0, 0 
#     sources, targets, sources_shape, targets_shape = [], [], [], []
    
#     p = padding()
    
#     for d in batch: 
#         s_frames = d['sources_shape'][0]
#         t_frames = d['targets_shape'][0]

#         if s_frames > max_frames: 
#             max_frames = s_frames  
#         if t_frames > max_frames: 
#             max_frames = t_frames 
    
#     for d in batch: # iterate over list elements len(batch) = batch_size 
#         s_frames, s_players = d['sources_shape'] # [frames, players, features]
#         t_frames, t_players = d['targets_shape'] # [frames, players, features]  
        
#         pad_s_players = p.max_source_players - s_players
#         pad_t_players = p.max_target_players - t_players
        
#         pad_s_frames = max_frames - s_frames 
#         pad_t_frames = max_frames - t_frames 
        
#         f = d['sources']
#         t = d['targets']
        
#         f = F.pad(f, (0, 0, 0, pad_s_players, 0, pad_s_frames), "constant", -1)
#         t = F.pad(t, (0, 0, 0, pad_t_players, 0, pad_t_frames), "constant", -1)

#         sources.append(f)
#         targets.append(t)
#         sources_shape.append((s_frames, s_players))
#         targets_shape.append((t_frames, t_players))
        
#     sources = torch.stack(sources)
#     targets = torch.stack(targets)
#     sources_shape = torch.tensor(sources_shape, dtype=sources.dtype, device=sources.device)
#     targets_shape = torch.tensor(targets_shape, dtype=targets.dtype, device=targets.device)
    
#     data = {
#         'sources': sources, 
#         'targets': targets,
#         'sources_shape': sources_shape,
#         'targets_shape': targets_shape
#     }
    
#     return data 