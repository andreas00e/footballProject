import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from typing import Dict, List, Tuple
from functools import partial
from omegaconf import OmegaConf

import torch 
from torchtyping import TensorType
from torch.utils.data import Dataset
from torch_geometric.data import Data

pd.set_option('future.no_silent_downcasting', True)
        
class GraphDataset(Dataset):     
    def __init__(self, data_dir: os.PathLike, scaling_path: os.PathLike, pos_path: os.PathLike, feature_config: dict, with_play_info: bool=True):
        self.data_dir = data_dir
        self.scaling_conf: OmegaConf = OmegaConf.load(scaling_path)
        self.pos_embeddings: Dict[str, float] = self._load_pos_embeddings(pos_path)
        self.feature_config = feature_config
        self.with_play_info = with_play_info

        self.plays: List[Tuple[str, int, int]] = []
        
        if self.with_play_info: 
            self.in_play_info: Dict[Tuple[str, int, int], Tuple[int, int, np.ndarray, int]] =  {}
            self.out_play_info: Dict[Tuple[str, int, int], int] =  {}
      
        self.edge_index_cache: Dict[int, TensorType["2, num_edges"]] = {}
        self._discover_plays() 
    
    def _discover_plays(self): 
        """ Method for creating a list of tuples containing a file name, a game_id, and a play_id
            to access files, games, and plays of interest during later data loading
            Optionally, additional information about how many frames the current play has, 
            how many players are present in the input or output, and the nfl_id of the players, 
            whose position should be forecasted can be added. 

        Raises:
            ValueError: Unexpected encounter of a csv file neither containing 'input' nor 'output' in its name.
        """
        csv_files = [file for file in os.listdir(self.data_dir) if file.endswith('.csv')]
        
        p_bar = tqdm(csv_files, colour='green')
        for file in p_bar:
            p_bar.set_description("Processing: {}".format(os.path.basename(file)))
            
            if not ('input' in file or 'output' in file):
                raise ValueError("Unexpected csv file name: {}. Expected 'input' or 'output' in filename.".format(file))  
        
            df_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(df_path, index_col=False) # df for one week of data
            games = df['game_id'].unique() # all games from one week 
            
            for game in games: 
                plays = df[df['game_id'] == game]['play_id'].unique() # id of every play from one game
                for play in plays: # save all frames from every play from one game and one week 
                    self.plays.append((file, game, play))
                    if self.with_play_info == True: 
                        play_frame = df[(df['game_id'] == game) & (df['play_id'] == play)]
                        n_frames = play_frame['frame_id'].nunique() 
                        n_in_players = play_frame['nfl_id'].nunique()                        
                        
                        if 'input' in file: 
                            player_to_predict = play_frame[play_frame['player_to_predict'] == 1]['nfl_id'].unique()
                            n_out_players = n_in_players - player_to_predict.shape[0]
                            self.in_play_info[(file, game, play)] = (n_frames, n_in_players, player_to_predict, n_out_players)
                        elif 'output' in file: 
                            self.out_play_info[(file, game, play)] = n_frames
                            
    def _get_edge_index(self, n_players: int) -> TensorType["2", "num_edges"]:
        if n_players not in self.edge_index_cache:
            A = torch.ones((n_players, n_players), dtype=torch.float32) - torch.eye(n_players, dtype=torch.float32)
            self.edge_index_cache[n_players] = torch.stack(A.nonzero(as_tuple=True), dim=0)
        return self.edge_index_cache[n_players]                  
        
    def _build_data(self, df: pd.DataFrame, file_type: str) -> List[Data]:
        data_list: List[Data] = []
        frames: np.ndarray = df['frame_id'].unique()
        n_players: np.ndarray = df['nfl_id'].nunique()
        edge_index = self._get_edge_index(n_players)

        for frame in frames:
            x = df[df['frame_id'] == frame]
            if file_type == 'in':
                x = x.apply(lambda v: self._normalize(v , self.scaling_conf, v.name) if v.name in self.feature_config['model']['norm'] else v)
            elif file_type == 'out': 
                x = x.apply(lambda v: self._normalize(v , self.scaling_conf, v.name) if v.name in ['x', 'y'] else v)
            x: TensorType["num_nodes", "num_features"] = torch.tensor(
                x.drop(columns=['nfl_id', 'frame_id'], errors='ignore').values, dtype=torch.float32
            )
            data_list.append(Data(x=x, edge_index=edge_index))
        return data_list

    @staticmethod
    def _load_pos_embeddings(path: os.PathLike) -> Dict[str, float]: 
        """ Method converting the literal name of a player to a 
            float value for later usage in model pipeline
        Args:
            path (os.PathLike): path to file holding all positions (str) that are 
                                present in the provided csv files
        Returns:
            Dict[str, float]: Dicitionary with the keys being the position name (str),
                              and the values being the the embeddings associated with 
                              their respective position (float)
        """
        data = {}
        with open(path) as f:
            for l in f: 
                key, val = l.strip().split(':')
                data[key] = float(val)
        return data 
    
    @staticmethod
    def _normalize(x: pd.Series, config_dict: dict, feature: str) -> pd.Series: 
        min_val, max_val = config_dict[feature]['min'], config_dict[feature]['max']
        return ((x - min_val) / (max_val - min_val)).astype(np.float32)  
    
    def __len__(self): 
        return len(self.plays)
        
    def __getitem__(self, index):       
        item = self.plays[index] # get key stored at index of key list 
        in_csv_file, game_id, play_id = item
        out_csv_file = in_csv_file.replace('input', 'output')
        
        input_frame = pd.read_csv(os.path.join(self.data_dir, in_csv_file), index_col=False)
        output_frame = pd.read_csv(os.path.join(self.data_dir, out_csv_file), index_col=False)
        
        input_frame = input_frame[(input_frame['game_id'] == game_id) & (input_frame['play_id'] == play_id)]
        output_frame = output_frame[(output_frame['game_id'] == game_id) & (output_frame['play_id'] == play_id)]
        # output_frame = output_frame[['frame_id', 'x', 'y']]
        
        input_features_of_interest = self.feature_config['loading'] + self.feature_config['model']['norm'] + self.feature_config['model']['no_norm']
        input_frame = input_frame[input_features_of_interest]
        input_frame[['player_to_predict', 'absolute_yardline_number', 'player_weight']] = np.float64(input_frame[['player_to_predict', 'absolute_yardline_number', 'player_weight']])
        input_frame = input_frame.replace({False: np.float64(0), True: np.float64(1), 'right': np.float64(0), 'left': np.float64(1), 'Defense': np.float64(0), 'Offense': np.float64(1)})
        input_frame['player_height'] = input_frame['player_height'].map(lambda x: np.float64(x.split('-')[0])*30.48+np.float64(x.split('-')[1])*2.54) # convert feet and inches to sane values (centimeters)
        input_frame['player_position'] = input_frame['player_position'].map(lambda x: np.float64(self.pos_embeddings[x])) 
        input_frame = input_frame.map(lambda x: x.astype(np.float32) if isinstance(x, np.float64) else x) # convert (np.)object to np.float32
    
        return self._build_data(input_frame, 'in'), self._build_data(output_frame, 'out')
    
        
class SequentialDataset(Dataset): 
    def __init__(self, data_path: os.PathLike, min_max_path: os.PathLike, pos_path: str, f_o_i: dict):
        self.data_path = data_path
        self.min_max: OmegaConf = OmegaConf.load(min_max_path)
        self.pos_emb = self.load_pos(pos_path)
        self.f_o_i = f_o_i
        self.files: List[os.PathLike] = os.listdir(self.data_path)
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