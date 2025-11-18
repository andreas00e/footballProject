import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial
from omegaconf import OmegaConf

import torch 
import torchtyping
from torch.utils.data import Dataset
from torch_geometric.data import Data

# pd.set_option('future.no_silent_downcasting', True) # TODO: Look-up what this does
        
class GraphDataset(Dataset):     
    def __init__(self, data_path: Path, min_max_path: Path, pos_path: Path, f_o_i: dict):
        self.data_path = data_path
        self.min_max: OmegaConf = OmegaConf.load(min_max_path)
        self.pos_emb = self.load_pos(pos_path)
        self.f_o_i = f_o_i
        self.files: List[Path] = os.listdir(self.data_path)
        self.length = 0
        self.input_plays, self.output_plays = {}, {} 
        self.item_list = [] # store all keys for later iteration 
        
        p_bar = tqdm(self.files, colour='green')
        for file in p_bar:
            p_bar.set_description("Processing: {}".format(os.path.basename(file)))

            path_for_frame = os.path.join(self.data_path, file)
            pd_frame = pd.read_csv(path_for_frame, index_col=False) # pandas data frame for one week of data
            games = pd_frame['game_id'].unique().tolist() # all games from one week 
            
            for game in games: 
                self.length += len(pd_frame[pd_frame['game_id'] == game]['frame_id'].unique().tolist()) # number of frames in one game
                plays = pd_frame[pd_frame['game_id'] == game]['play_id'].unique().tolist() # id of every play from one game
                for play in plays: # save all frames from every play from one game and one week 
                    if 'input' in file: 
                        self.input_plays[(file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist() 
                    elif 'output' in file: 
                        self.output_plays[(file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist()
                    else:
                        raise ValueError(f"Unexpected CSV file name: {file}. Expected 'input' or 'output' in filename.")                    
                    
        self.item_list = list(self.input_plays.keys())  
    
    def __len__(self): 
        return len(self.item_list)
        
    def __getitem__(self, index):       
        item = self.item_list[index] # get key stored at index of key list 
        in_csv_file, game_id, play_id = item
        out_csv_file = in_csv_file.replace('input', 'output')
    
        input_frame_ids = self.input_plays[item] # frame ids for every frame in one play from one game and one week
        output_frame_ids = self.output_plays[(out_csv_file, game_id, play_id)]
        
        input_frame = pd.read_csv(os.path.join(self.data_path, in_csv_file), index_col=False)
        output_frame = pd.read_csv(os.path.join(self.data_path, out_csv_file), index_col=False)
        
        input_frame = input_frame[(input_frame['game_id'] == game_id) & (input_frame['play_id'] == play_id)] # filter for current game and  current play
        output_frame = output_frame[(output_frame['game_id'] == game_id) & (output_frame['play_id'] == play_id)]
        output_frame = output_frame[['frame_id', 'x', 'y']] # filter for output features of interest 
        
        input_frames = input_frame['frame_id'].unique().tolist() # TODO: Think about putting the number of frames in the item list as well 
        output_frames = output_frame['frame_id'].unique().tolist()
        n_frames_output = output_frame['frame_id'].unique().tolist() 
        
        input_features_of_interest = self.f_o_i['loading'] + self.f_o_i['model']['norm'] + self.f_o_i['model']['no_norm'] # get features of interest from yaml file
        input_frame = input_frame[input_features_of_interest] # filter for input features of interest
        input_frame = input_frame.replace({False: np.float32(0), True: np.float32(1), 'right': np.float32(0), 'left': np.float32(1), 'Defense': np.float32(0), 'Offense': np.float32(1)})
        input_frame['player_height'] = input_frame['player_height'].map(lambda x: np.float32(x.split('-')[0])*30.48+np.float32(x.split('-')[1])*2.54) # convert feet and inches to sane values (centimeters)
        input_frame['player_position'] = input_frame['player_position'].map(lambda x: np.float32(self.pos_emb[x])) # convert literal positions to numeric positions 
        input_frame = input_frame.map(lambda x: x.astype(np.float32) if isinstance(x, object) else x) # convert (np.)object to np.float32
        
        n_players = len(input_frame['nfl_id'].unique().tolist())
        
        func = partial(self.normalize, config_dict=self.min_max)
        input_frame = (
            input_frame[input_frame['frame_id'].isin(input_frame_ids)]
            .apply(
                lambda x: func(x=x, feature=x.name)
                if x.name in self.f_o_i['model']['norm'] else x,
                axis=0
            )
        )
        
        output_frame = (
            output_frame[output_frame['frame_id'].isin(output_frame_ids)]
            .apply(
                lambda x: func(x=x, feature=x.name)
                if x.name in ['x', 'y'] else x,
                axis=0
            )
        )
        
        input_frame_= input_frame.drop(columns=self.f_o_i['loading'])
        output_frame = output_frame.drop(columns='frame_id')
        # input_frame = [x.drop(columns=self.f_o_i['loading']) for _, x in input_frame.groupby(by='frame_id')]
        
        node_in_features = {frame: torch.tensor(input_frame[input_frame['frame_id'] == frame].values, dtype=torch.float32) for frame in input_frames} # -> [frames, nodes, features]
        node_out_features = {frame: torch.tensor(output_frame[output_frame['frame_id'] == frame].values, dtype=torch.float32) for frame in output_frames} # -> [frames, nodes, features]
        
        features = [Data(x=node_in_features[frame], edge_index=self._graph_connectivity(n_players), y=node_in_features[frame]) for frame in input_frames]
        targets = [Data(x=node_out_features[frame], edge_index=self._graph_connectivity(n_players), y=node_out_features[frame]) for frame in output_frames]

        return features, targets
    
    @staticmethod
    def load_pos(path: Path) -> Dict: 
        data = {}
        with open(path) as f:
            for l in f: 
                x = l.strip().split(':')
                pos, emb = x[0], x[1]
                data[pos] = emb
        return data   
    
    @staticmethod
    def normalize(x: pd.Series, config_dict: dict, feature: str) -> pd.Series: 
        min: float = config_dict[feature]['min']
        max: float = config_dict[feature]['max']
        
        return np.float32((x-min)/(max-min)) 
    
    @staticmethod
    def _graph_connectivity(n_players: int): 
        """
        Create edge_index tensor for torch_geometric.data.Data.
        Returns all directed edges between different nodes (no self-loops).

        Args:
            n_players (int): Number of nodes.

        Returns:
            Tensor: torch.float32 tensor of shape [2, num_edges] containing edge index pairs.
        """
        A = torch.ones((n_players, n_players), dtype=torch.float32)-torch.eye(n_players, dtype=torch.float32)
        B = A.nonzero(as_tuple=True)
        C = torch.stack(B, dim=0)

        return C 
        
class SequentialDataset(Dataset): 
    def __init__(self, data_path: Path, min_max_path: Path, pos_path: Path, f_o_i: dict):
        self.data_path = data_path
        self.min_max: OmegaConf = OmegaConf.load(min_max_path)
        self.pos_emb = self.load_pos(pos_path)
        self.f_o_i = f_o_i
        self.files: List[Path] = os.listdir(self.data_path)
        self.length = 0
        self.input_plays, self.output_plays = {}, {} 
        self.item_list = [] # store all keys for later iteration 
        
        p_bar = tqdm(self.files, colour='green')
        for file in p_bar:
            p_bar.set_description("Processing: {}".format(os.path.basename(file)))

            path_for_frame = os.path.join(self.data_path, file)
            pd_frame = pd.read_csv(path_for_frame, index_col=False) # pandas data frame for one week of data
            games = pd_frame['game_id'].unique().tolist() # all games from one week 
            
            for game in games: 
                self.length += len(pd_frame[pd_frame['game_id'] == game]['frame_id'].unique().tolist()) # number of frames in one game
                plays = pd_frame[pd_frame['game_id'] == game]['play_id'].unique().tolist() # id of every play from one game
                for play in plays: # save all frames from every play from one game and one week 
                    if 'input' in file: 
                        self.input_plays[(file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist() 
                    elif 'output' in file: 
                        self.output_plays[(file, game, play)] = pd_frame[pd_frame['play_id'] == play]['frame_id'].unique().tolist()
                    else:
                        raise ValueError(f"Unexpected CSV file name: {file}. Expected 'input' or 'output' in filename.")                    
                    
        self.item_list = list(self.input_plays.keys())  
    
    def __len__(self): 
        return self.length
        
    def __getitem__(self, index):       
        item = self.item_list[index] # get key stored at index of key list 
        in_csv_file, game_id, play_id = item
        out_csv_file = in_csv_file.replace('input', 'output')
    
        input_frame_ids = self.input_plays[item] # frame ids for every frame in one play from one game and one week
        output_frame_ids = self.output_plays[(out_csv_file, game_id, play_id)]
        
        input_frame = pd.read_csv(os.path.join(self.data_path, in_csv_file), index_col=False)
        output_frame = pd.read_csv(os.path.join(self.data_path, out_csv_file), index_col=False)
        
        input_frame = input_frame[(input_frame['game_id'] == game_id) & (input_frame['play_id'] == play_id)] # filter for current game and  current play
        output_frame = output_frame[(output_frame['game_id'] == game_id) & (output_frame['play_id'] == play_id)]
        output_frame = output_frame[['frame_id', 'x', 'y']] # filter for output features of interest 
         
        input_features_of_interest = self.f_o_i['loading'] + self.f_o_i['model']['norm'] + self.f_o_i['model']['no_norm'] # get features of interest from yaml file
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
        print(input_frame)
        input_frame = [x.drop(columns=self.f_o_i['loading']) for _, x in input_frame.groupby(by='frame_id')]
        print(input_frame[0])
        exit()

        # input_frame = [torch.from_numpy(x.drop(columns=self.f_o_i['loading']).values) for _, x in input_frame.groupby(by='frame_id')]

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