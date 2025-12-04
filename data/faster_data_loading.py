import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from typing import Dict, List, Tuple, Union
from omegaconf import OmegaConf

import torch 
from torchtyping import TensorType
from torch.utils.data import Dataset
from torch_geometric.data import Data

pd.set_option('future.no_silent_downcasting', True)
        
class PlayDataset(Dataset):     
    def __init__(self, data_dir: os.PathLike, scaling_path: os.PathLike, pos_path: os.PathLike, feature_config: dict, data_type: str):
        self.data_dir = data_dir
        self.scaling_conf: OmegaConf = OmegaConf.load(scaling_path)
        self.pos_embedds: Dict[str, float] = self._load_pos_embedds(pos_path) 
        self.feature_config = feature_config
        self.data_type = data_type

        self.df_cache: Dict = {}
        self.item_list: List[Tuple[os.PathLike, str, str]] = []
        
        if self.data_type == 'graph':
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
        csv_files = [file for file in os.listdir(self.data_dir) if file.endswith('.csv') if 'input' in file]
        
        p_bar = tqdm(csv_files, colour='green')
        for file in p_bar:
            file_type = None
            p_bar.set_description("Processing: {}".format(os.path.basename(file)))
             
            df_path = os.path.join(self.data_dir, file)
            
            if 'input' in file: 
                cols = self.feature_config.loading+self.feature_config.model.norm+self.feature_config.model.no_norm
                file_type = 'input'
            elif 'output' in file: 
                cols = self.feature_config.loading+['x', 'y']
                file_type = 'output'
            else: 
                raise ValueError("Unexpected csv file name: {}. Expected 'input' or 'output' in filename.".format(file))  

            df = pd.read_csv(df_path, index_col=False, usecols=cols) # one week of data
            if 'input' in file:
                df = df.replace({False: 0.0, True: 1.0, 'right': 0.0, 'left': 1.0, 'Defense': 0.0, 'Offense': 1.0})
                df['player_height'] = df['player_height'].map(lambda x: np.float32(x.split('-')[0])*30.48+np.float32(x.split('-')[1])*2.54) # convert feet and inches to sane values (centimeters)
                df['player_position'] = df['player_position'].map(lambda x: np.float32(self.pos_embedds[x])) 
                df = df.astype(np.float32)
                games = df['game_id'].unique() 
            
            df = self._normalize(df, file_type)
                        
            for game in games: # one game of one week of data
                plays = df[df['game_id'] == game]['play_id'].unique() 
                for play in plays: # one play of one game of one week of data
                    play_frame = df[(df['game_id'] == game) & (df['play_id'] == play)]                
                    
                    data = self._build_data(play_frame, file_type, data_type=self.data_type)
                    
                    self.df_cache[(file, game, play)] = data
        
        self.item_list = list(self.df_cache.keys())
                            
    def _get_edge_index(self, n_players: int) -> TensorType["2", "num_edges"]:
        if n_players not in self.edge_index_cache:
            A = torch.ones((n_players, n_players), dtype=torch.float32) - torch.eye(n_players, dtype=torch.float32)
            self.edge_index_cache[n_players] = torch.stack(A.nonzero(as_tuple=True), dim=0)
        return self.edge_index_cache[n_players]                  
        
    def _build_data(self, df: pd.DataFrame, file_type: str, data_type: str) -> List[Data]:
        data_list: List[Data] = []
        n_players: np.ndarray = df['nfl_id'].nunique()

        df = df.astype(np.float64)
        if self.data_type == 'graph': 
            edge_index = self._get_edge_index(n_players)
            
        for _, f in df.groupby('frame_id'): 
            f = f.drop(columns=['game_id', 'play_id', 'nfl_id', 'frame_id'], errors='ignore')
            f = torch.tensor(f.values, dtype=torch.float32)
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
            cols = list(self.feature_config['model']['norm']) 
        elif file_type == 'output': 
            cols = ['x', 'y']
        
        X = df.loc[:, cols].values.astype(np.float32)
        min = np.array([self.scaling_conf[f]['min'] for f in cols], dtype=np.float32)
        max = np.array([self.scaling_conf[f]['max'] for f in cols], dtype=np.float32)
         
        df.loc[:, cols] = ((X - min) / (max - min))
        return df 

    @staticmethod
    def _load_pos_embedds(path: os.PathLike) -> Dict[str, float]: 
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
        return len(self.item_list)
        
    def __getitem__(self, index) -> Dict[TensorType, TensorType]:       
        file, game, play = self.item_list[index]
        if 'input' in file:
            in_file = file
            out_file = file.replace('input', 'output')
        else: 
            in_file = file.replace('output', 'input')
            out_file = file
             
        sources = self.df_cache[(in_file, int(game), int(play))] 
        targets = self.df_cache[(out_file, int(game), int(play))]            
        
        data = {
            'sources': sources, 
            'targets': targets,
            'sources_shape': sources.shape[:2],
            'targets_shape': targets.shape[:2]
            }
        
        return data