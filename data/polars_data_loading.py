import os 
import polars as pl
from tqdm import tqdm 
from typing import Dict, List, Tuple, Union
from omegaconf import OmegaConf

import torch 
from torchtyping import TensorType
from torch.utils.data import Dataset
from torch_geometric.data import Data
        
class PlayDataset(Dataset):     
    def __init__(self, data_dir: os.PathLike, scaling_path: os.PathLike, pos_path: os.PathLike, feature_config: dict, data_type: str):
        self.data_dir = data_dir
        self.files = [file for file in os.listdir(self.data_dir) if file.endswith(".csv")]
        
        self.scaling_conf: OmegaConf = OmegaConf.load(scaling_path)
        self.pos_embedds: Dict[str, float] = self._load_pos_embedds(pos_path) 
        self.feature_config = feature_config
        self.data_type = data_type
        self.plays_abc = []
        
        if self.data_type == "graph":
            self.edge_index_cache: Dict[int, TensorType["2, num_edges"]] = {}

        self.df_cache: Dict = {}
        self.item_list: List[Tuple[os.PathLike, str, str]] = []
            
        self._discover_plays() 
    
    def _discover_plays(self) -> None:
        p_bar = tqdm(self.files, colour="green")
        
        for file in p_bar:
            print("FILE")
            p_bar.set_description("Processing: {}".format(os.path.basename(file)))
            
            file_type = None
            df_path = os.path.join(self.data_dir, file)
            
            if "input" in file: 
                file_type = "input"
                cols = list(self.feature_config.used)
            elif "output" in file: 
                file_type = "output"
                cols = list(self.feature_config.loading) + ["x", "y"]
            else: 
                raise ValueError("Unexpected csv file name: {}. Expected \"input\" or \"output\" in filename.".format(file))  
                         
            df = pl.read_csv(df_path, columns=cols) 
                                
            if "input" in file:
                df = df.with_columns([
                    pl.when(pl.col("player_to_predict") == False).then(0.0).otherwise(1.0).alias("player_to_predict"),
                    pl.when(pl.col("play_direction") == "right").then(0.0).otherwise(1.0).alias("play_direction"),
                    pl.when(pl.col("player_side") == "Defense").then(0.0).otherwise(1.0).alias("player_side"),
                    
                    pl.col("player_position").replace_strict(self.pos_embedds, default=-1).alias("player_position"),
                    (pl.col("player_height").str.split_exact("-", 2).struct.field("field_0").cast(pl.Float32) * 30.48 +
                    pl.col("player_height").str.split_exact("-", 2).struct.field("field_1").cast(pl.Float32) * 2.54
                    ).alias("player_height")
                ])
                            
            df = self._normalize(df, file_type)
            
            games = df["game_id"].unique()
            for game in games: # one game of one week of data
                plays = df.filter(pl.col("game_id").cast(pl.Int64) == game)["play_id"].unique()
                for play in plays:
                    frame = df.filter((pl.col("game_id").cast(pl.Int64) == game) & (pl.col("play_id") == play))
                    data = self._build_data(frame, self.data_type)
                    self.df_cache[(file, game, play)] = data  
        
        self.item_list = list(self.df_cache.keys())
                            
    def _get_edge_index(self, n_players: int) -> TensorType["2", "num_edges"]:
        if n_players not in self.edge_index_cache:
            A = torch.ones((n_players, n_players), dtype=torch.float32) - torch.eye(n_players, dtype=torch.float32)
            self.edge_index_cache[n_players] = torch.stack(A.nonzero(as_tuple=True), dim=0)
        return self.edge_index_cache[n_players]                  
        
    def _build_data(self, frame: pl.DataFrame, data_type: str) -> Union[List[TensorType] | List[Data]]:
        n_players: int = frame["nfl_id"].n_unique()
        n_frames: int = frame["frame_id"].n_unique()
        data_list = [None for _ in range(n_frames)]

        if self.data_type == "graph": 
            edge_index = self._get_edge_index(n_players)
           
        for i, f in enumerate(frame.partition_by("frame_id")): 
            f = f.drop(["game_id", "play_id", "nfl_id", "frame_id"])
            if data_type == "sequential": 
                data_list[i] = f.to_torch(dtype=pl.Float32) 
            elif data_type == "graph": 
                data_list.append(Data(x=f, edge_index=edge_index))
                
        return data_list if data_type == "graph" else torch.stack(data_list, dim=0)
    
    def _normalize(self, df: pl.DataFrame, file_type: str) -> pl.DataFrame:
        """Method normalizing, in the case of an input file, every in the respective yaml file 
           stated variable, and in the case of an output file, the x-coordinates and the y-coordinates

        Args:
            df (pd.DataFrame): pd.DataFrame whose values are to be normalized
            file_type (str): Informing the method whether it was given an input file or an output file 

        Returns:
            pd.DataFrame: pd.DataFrame with the in the resepctive yaml file 
            or only the x-coordinate and y-coordinate normalized columns
        """
        if file_type == "input": 
            scaling_conf = self.scaling_conf
        if file_type == "output": 
            scaling_conf = {k: self.scaling_conf[k] for k in ["x", "y"]}

        df = df.with_columns((pl.col(k) - v["min"]) / (v["max"] -  v["min"]) for k, v in scaling_conf.items())
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