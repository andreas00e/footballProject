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
        
class PlayDataset(Dataset):     
    def __init__(self, data_dir: os.PathLike, scaling_path: os.PathLike, pos_path: os.PathLike, feature_config: dict, data_type: str, with_play_info: bool=True):
        self.data_dir = data_dir
        self.scaling_conf: OmegaConf = OmegaConf.load(scaling_path)
        self.pos_embeddings: Dict[str, float] = self._load_pos_embeddings(pos_path)
        self.feature_config = feature_config
        self.data_type = data_type
        self.with_play_info = with_play_info

        self.plays: List[Tuple[str, int, int]] = []
        self.df_cache: Dict = {}
        
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
            self.df_cache[file] = df
            
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
        
    def _build_data(self, df: pd.DataFrame, file_type: str, data_type: str) -> List[Data]:
        data_list: List[Data] = []
        n_players: np.ndarray = df['nfl_id'].nunique()
        edge_index = self._get_edge_index(n_players)
        
        df = self._normalize(df, file_type)
        
        for _, f in df.groupby('frame_id'): 
            f = torch.tensor(f.drop(columns=['nfl_id', 'frame_id'], errors='ignore').values, dtype=torch.float32)
            if data_type == 'sequential': 
                data_list.append(f) 
            elif data_type == 'graph': 
                data_list.append(Data(x=f, edge_index=edge_index))
        return data_list if data_type == 'graph' else torch.stack(data_list, dim=0)
    
    def _normalize(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Method normalizing, in the case of an input file, every in the respective yaml file 
           stated variable, and in the case of an output file, the x-coordinates and the y-coordinates

        Args:
            df (pd.DataFrame): pd.DataFrame whose values are to be normalized
            file_type (str): Informing the method whether it was given an input file or an output file 

        Returns:
            pd.DataFrame: pd.DataFrame with the in the resepctive yaml file 
            or only the x-coordinate and y-coordinate normalized columns
        """
        if file_type == 'input': 
            cols = self.feature_config['model']['norm'] 
        elif file_type == 'output': 
            cols = ['x', 'y']
            
        X = df.loc[:, cols].values.astype(np.float32)

        min = np.array([self.scaling_conf[f]['min'] for f in cols], dtype=np.float32)
        max = np.array([self.scaling_conf[f]['max'] for f in cols], dtype=np.float32)

        df.loc[:, cols] = (X - min) / (max - min)
        return df 

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

    def __len__(self): 
        return len(self.plays)
        
    def __getitem__(self, index):       
        item = self.plays[index] # get keys stored at index of key list 
        file, game_id, play_id = item   
        if 'input' in file:
            in_file = file 
            out_file = file.replace('input', 'output')
        else: 
            in_file = file.replace('output', 'input')
            out_file = file
             
        input_frame = self.df_cache[in_file] # TODO: Load only those columns that will be accessed later
        output_frame = self.df_cache[out_file]
                
        input_frame = input_frame[(input_frame['game_id'] == game_id) & (input_frame['play_id'] == play_id)]
        output_frame = output_frame[(output_frame['game_id'] == game_id) & (output_frame['play_id'] == play_id)]
        output_frame = output_frame[['nfl_id', 'frame_id', 'x', 'y']]
        
        input_features_of_interest = self.feature_config['loading'] + self.feature_config['model']['norm'] + self.feature_config['model']['no_norm']
        input_frame = input_frame[input_features_of_interest]
        input_frame = input_frame.replace({False: 0, True: 1, 'right': 0, 'left': 1, 'Defense': 0, 'Offense': 1})
        to_np_float_64 = ['player_weight', 'absolute_yardline_number', 'player_to_predict', 'play_direction', 'player_side']
        input_frame[to_np_float_64] = input_frame[to_np_float_64].astype(np.float64)
        input_frame['player_height'] = input_frame['player_height'].map(lambda x: np.float64(x.split('-')[0])*30.48+np.float64(x.split('-')[1])*2.54) # convert feet and inches to sane values (centimeters)
        input_frame['player_position'] = input_frame['player_position'].map(lambda x: np.float64(self.pos_embeddings[x])) 
        return self._build_data(input_frame, file_type='input', data_type=self.data_type), self._build_data(output_frame, file_type='output', data_type=self.data_type)