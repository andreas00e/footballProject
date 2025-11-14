import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from pathlib import Path
from typing import List, Dict
from functools import partial
from omegaconf import OmegaConf

import torch 
from torch.utils.data import Dataset
from torch_geometric.data import Data

pd.set_option('future.no_silent_downcasting', True) # TODO: Look-up what this does

def normalize(x: pd.Series, config_dict: dict, feature: str) -> pd.Series: 
    min: float = config_dict[feature]['min']
    max: float = config_dict[feature]['max']
    
    return np.float32((x-min)/(max-min))

def load_pos(path: Path): 
    data = dict()
    with open(path) as f:
        for l in f: 
            x = l.strip().split(':')
            pos, emb = x[0], x[1]
            data[pos] = emb
    return data          
        
class GraphDataset(Dataset):     
    def __init__(self, path):
        self.path = path
        self.csv_files = os.listdir(path)
        self.length = 0
        self.input_plays = {}       
        self.output_plays = {} 
        self.item_list = [] # store all keys for later iteration 
        self.features_of_interest = ['nfl_id', 'frame_id', 'player_to_predict', 'play_direction', 'absolute_yardline_number', 'player_height', 'player_weight', 'player_side', 'x', 'y', 's', 'a', 'dir', 'o'] # TODO: Create a config file to which this part can be moved to 

        for csv_file in tqdm(self.csv_files, colour='green'):
            path_frame = os.path.join(self.path, csv_file)
            pd_frame = pd.read_csv(path_frame) # pandas data frame of current week
            games = pd_frame['game_id'].unique().tolist() # all games of the current week 
            
            for game in games: 
                self.length += len(pd_frame[pd_frame['game_id'] == game]['frame_id'].unique().tolist())
                plays = pd_frame[pd_frame['game_id'] == game]['play_id'].unique().tolist()
                for play in plays:
                    if 'input' in csv_file: 
                        self.input_plays[(csv_file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist() # save all frames of current game and current week 
                    elif 'output' in csv_file: 
                        self.output_plays[(csv_file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist()
                    else:
                        raise ValueError(f"Unexpected CSV file name: {csv_file!r}. Expected 'input' or 'output' in filename.")                    
                    
        self.item_list = list(self.input_plays.keys())  
    
    def __len__(self): 
        return self.length
        
    def __getitem__(self, index):         
        item = self.item_list[index] # get key stored at position index of key list 
        in_csv_file, game_id, play_id = item
        out_csv_file = in_csv_file.replace('input', 'output')
    
        input_frame_ids = self.input_plays[item] 
        output_frame_ids = self.output_plays[(out_csv_file, game_id, play_id)]
        output_frame = pd.read_csv(os.path.join(self.path, out_csv_file))
        output_frame = output_frame[(output_frame['game_id'] == game_id) & (output_frame['play_id'] == play_id)]
        
        input_frame = pd.read_csv(os.path.join(self.path, in_csv_file))
        input_frame = input_frame[(input_frame['game_id'] == game_id) & (input_frame['play_id'] == play_id)]
        input_frame = input_frame[self.features_of_interest].copy()
        input_frame = (input_frame.replace({False: 0, True: 1, 'right': 0, 'left': 1, 'Defense': 0, 'Offense': 1}).infer_objects(copy=False))
        input_frame['player_height'] = input_frame['player_height'].apply(lambda x: float(x.split('-')[0])*30.48+float(x.split('-')[1])*2.54) # convert feet and inches to sane values (centimeters)

        features = []
        for frame_id in input_frame_ids: 
            current_frame = input_frame[(input_frame['frame_id'] == frame_id)]
            features.append(torch.tensor(current_frame.values, dtype=torch.float32))
            
        targets = []
        for frame_id in output_frame_ids: 
            current_frame = output_frame[(output_frame['frame_id'] == frame_id)]
            targets.append(torch.tensor(current_frame.values, dtype=torch.float32))
            
        exit()
        
        node_labels = input_frame['nfl_id'].unique().tolist() # one node is one player present in the current play         
        n = len(node_labels) # number of nodes in the current graph
        source = torch.arange(n, dtype=torch.long).repeat_interleave(n) # repeat each element
        target = torch.arange(n, dtype=torch.long).repeat(n) # repeat tensor 
        mask = source != target
        
        edge_indices = [torch.stack([source[mask], target[mask]], dim=0)]*len(input_frame_ids)
        edge_weights = [torch.ones(len(node_labels)).unsqueeze(1)]*len(input_frame_ids)
        
        snapshots = Data()
        
        snapshots = DynamicGraphTemporalSignal(edge_indices=edge_indices, edge_weights=edge_weights, features=features, targets=targets)
        
        return snapshots
        
class SequentialDataset(Dataset): 
    def __init__(self, data_path: Path, min_max_path: Path, pos_path: Path, f_o_i: dict):
        self.data_path = data_path
        self.min_max: OmegaConf = OmegaConf.load(min_max_path)
        self.pos_emb = load_pos(pos_path)
        self.f_o_i = f_o_i
        self.csv_files: List[Path] = os.listdir(self.data_path)
        self.length = 0
        self.input_plays = dict()      
        self.output_plays = dict() 
        self.item_list = list() # store all keys for later iteration 
        
        for csv_file in tqdm(self.csv_files, colour='green'):
            path_frame = os.path.join(self.data_path, csv_file)
            pd_frame = pd.read_csv(path_frame) # pandas data frame for one week
            games = pd_frame['game_id'].unique().tolist() # all games from one week 
            
            for game in games: 
                self.length += len(pd_frame[pd_frame['game_id'] == game]['frame_id'].unique().tolist()) # number of frames in one game
                plays = pd_frame[pd_frame['game_id'] == game]['play_id'].unique().tolist() # id of every play from one game
                for play in plays: 
                    if 'input' in csv_file: 
                        self.input_plays[(csv_file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist() # save all frames from one game and one week 
                    elif 'output' in csv_file: 
                        self.output_plays[(csv_file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist()
                    else:
                        raise ValueError(f"Unexpected CSV file name: {csv_file}. Expected 'input' or 'output' in filename.")                    
                    
        self.item_list = list(self.input_plays.keys())  
    
    def __len__(self): 
        return self.length
        
    def __getitem__(self, index):       
        item = self.item_list[index] # get key stored at index of key list 
        in_csv_file, game_id, play_id = item
        out_csv_file = in_csv_file.replace('input', 'output')
    
        input_frame_ids = self.input_plays[item] 
        output_frame_ids = self.output_plays[(out_csv_file, game_id, play_id)]
        
        input_frame = pd.read_csv(os.path.join(self.data_path, in_csv_file))
        output_frame = pd.read_csv(os.path.join(self.data_path, out_csv_file))
        
        input_frame = input_frame[(input_frame['game_id'] == game_id) & (input_frame['play_id'] == play_id)] # filter for game and play in input
        output_frame = output_frame[(output_frame['game_id'] == game_id) & (output_frame['play_id'] == play_id)] # filter for game and play in output 
        output_frame = output_frame[['frame_id', 'x', 'y']] # filter for output features of interest 
         
        input_features_of_interest = self.f_o_i['loading']+self.f_o_i['model']['norm']+self.f_o_i['model']['no_norm'] # prepare features of interest
        input_frame = input_frame[input_features_of_interest] # filter for input features of interest
        input_frame = input_frame.replace({False: np.float32(0), True: np.float32(1), 'right': np.float32(0), 'left': np.float32(1), 'Defense': np.float32(0), 'Offense': np.float32(1)})
        input_frame['player_height'] = input_frame['player_height'].apply(lambda x: np.float32(x.split('-')[0])*30.48+np.float32(x.split('-')[1])*2.54) # convert feet and inches to sane values (centimeters)
        input_frame['player_position'] = input_frame['player_position'].apply(lambda x: np.float32(self.pos_emb[x])) # convert literal positions to numeric positions 
        input_frame = input_frame.apply(lambda x: x.astype(np.float32) if isinstance(x, object) else x) # convert (np.)object to np.float32
        
        func = partial(normalize, config_dict=self.min_max)
        input_frame = (
            input_frame[input_frame['frame_id'].isin(input_frame_ids)]
            .apply(
                lambda x: func(x=x, feature=x.name)
                if x.name in self.f_o_i['model']['norm'] else x,
                axis=0
            )
        )
        input_frame = [torch.from_numpy(x.drop(columns=self.f_o_i['loading']).values) for _, x in input_frame.groupby(by='frame_id')]

        output_frame = (
            output_frame[output_frame['frame_id'].isin(output_frame_ids)]
            .apply(
                lambda x: func(x=x, feature=x.name)
                if x.name in ['x', 'y'] else x,
                axis=0
            )
        )
        
        output_frame = [torch.from_numpy(x.drop(columns='frame_id').values) for _, x in output_frame.groupby(by='frame_id')]
        
        features = torch.stack(input_frame, dim=0)
        targets = torch.stack(output_frame, dim=0)

        return features, targets