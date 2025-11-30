import pygmtools as pygm
pygm.set_backend('pytorch')

from typing import Dict

import torch
import torch.nn.functional as F 
from torchtyping import TensorType

max_s_players = 17 # XXX: If those numbers are bigger in one of the test files, this wouldn't work ...
max_t_players = 9

def collate_fn(batch) -> Dict[str, TensorType['*']]: 
    max_frames = 0
    s_frames, t_frames = 0, 0 
    sources, targets, sources_shape, targets_shape = [], [], [], []
    
    for d in batch: 
        s_frames = d['sources_shape'][0]
        t_frames = d['targets_shape'][0]

        if s_frames > max_frames: 
            max_frames = s_frames  
        if t_frames > max_frames: 
            max_frames = t_frames 
    
    for d in batch: # iterate over list elements len(batch) = batch_size 
        s_frames, s_players = d['sources_shape'] # [frames, players, features]
        t_frames, t_players = d['targets_shape'] # [frames, players, features]  
        
        pad_s_players = max_s_players - s_players
        pad_t_players = max_t_players - t_players
        
        pad_s_frames = max_frames - s_frames 
        pad_t_frames = max_frames - t_frames 
        
        f = d['sources']
        t = d['targets']
        
        f = F.pad(f, (0, 0, 0, pad_s_players, 0, pad_s_frames), "constant", 0)
        t = F.pad(t, (0, 0, 0, pad_t_players, 0, pad_t_frames), "constant", 0)

        sources.append(f)
        targets.append(t)
        sources_shape.append((s_frames, s_players))
        targets_shape.append((t_frames, t_players))
        
    sources = torch.stack(sources)
    targets = torch.stack(targets)
    sources_shape = torch.tensor(sources_shape, dtype=sources.dtype, device=sources.device)
    targets_shape = torch.tensor(targets_shape, dtype=targets.dtype, device=targets.device)
    
    data = {
        'sources': sources, 
        'targets': targets,
        'sources_shape': sources_shape,
        'targets_shape': targets_shape
    }
    
    return data 

def hungarian_matching(target: TensorType["batch*t_frames", "t_players", "t_features"], prediction: TensorType["batch*t_frames", "t_players", "t_features"]) -> \
    TensorType['batch*t_frames', 't_players', 't_player']: 
    target_x = target[:, :, 0:1]
    target_y = target[:, :, 1:]
    prediction_x = prediction[:, :, 0:1]
    prediction_y = prediction[:, :, 1:]
    
    ones = torch.ones_like(target_x, dtype=target_x.dtype, device=target.device)
        
    target_x_view = target_x  @ torch.transpose(ones, dim0=-2, dim1=-1)
    target_y_view = ones @ torch.transpose(target_y, dim0=-2, dim1=-1) 
    prediction_x_view = prediction_x  @ torch.transpose(ones, dim0=-2, dim1=-1)
    prediciton_y_view = ones @ torch.transpose(prediction_y, dim0=-2, dim1=-1) 
 
    x_cost = torch.pow(input=(target_x_view - prediction_x_view), exponent=2)
    y_cost = torch.pow(input=(target_y_view - prediciton_y_view), exponent=2)
    
    cost = 1/2 * (x_cost + y_cost) 
    
    assignment = pygm.hungarian(cost)
    return assignment