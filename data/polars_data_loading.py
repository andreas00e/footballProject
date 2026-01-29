import os 
import numpy as np 
import polars as pl
from tqdm import tqdm 
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Union

import torch 
from torchtyping import TensorType
from torch.utils.data import Dataset
from torch_geometric.data import Data

def geometric_output_features(groups: np.ndarray) -> np.ndarray: 
    """ A method that augments the two coordinates given in the output to the same 
        number of features as in the input to faciliate later learning
    Args:
        groups (np.ndarray): The array containing the x- and y-coordinates from every player of one frame 

    Returns:
        np.ndarray: The augmented array 
    """
    x = groups[:, :, 0]
    y = groups[:, :, 1]
    theta = np.arctan2(y, x)
    x_x = x*x 
    y_y = y*y 
    
    features = [
        x, 
        y, 
        np.sqrt(x), 
        theta, 
        x_x, 
        y_y, 
        x*y, 
        np.abs(x), 
        np.abs(y), 
        np.sin(theta), 
        np.cos(theta), 
        np.sqrt(x_x+y_y), # every player's distance to the origin 
        np.ones_like(x), # bias
        x-x+1
        ]
        
    return np.stack(features, axis=-1)
         
class PlayDataset(Dataset):     
    def __init__(self, data_dir: os.PathLike, scaling_path: os.PathLike, pos_path: os.PathLike, feature_config: dict, data_type: str, mode: str):
        self.mode = mode
        self.data_dir = data_dir
        if os.path.isdir(data_dir): 
            self.files = [file for file in os.listdir(self.data_dir) if file.endswith(".csv")]
        elif os.path.isfile(data_dir): 
            self.files = [data_dir]
        
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
        """ A method for loading, preprocessing, and normalizing polars dataframes representing the given CSV files.
            The resulting dataframe is then saved as the value of the dictionary, with the corresponding key being 
            a triplet of the file name, the game ID, and the play ID. The item list serves as the iterable to be accessed 
            when the dataset is in use for, e.g., training the model.

        Raises:
            ValueError: If neither "input" nor "output" unexpectedly is not part of teh currently loaded file name
        """

        p_bar = tqdm(self.files, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]", colour="green")
        for file in p_bar:
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
                    (pl.col("player_height").str.slice(0,1).cast(pl.Float32) * 30.48 +
                    pl.col("player_height").str.slice(2,2).cast(pl.Float32) * 2.54
                    ).alias("player_height")
                ])
                
            df = self._normalize(df=df, file_type=file_type)
                            
            for name, frame in df.group_by(["game_id", "play_id"]): 
                game, play = name 
                frame = frame.drop(["game_id", "play_id", "nfl_id"])
                
                if self.data_type == "graph":
                    self.df_cache[(file, game, play)] = self._build_data(df=frame, rows=frame.shape[0], file_type=file_type)

                else: 
                    data = self._build_data(df=frame, rows=frame.shape[0], file_type=file_type)
                    self.df_cache[(file, game, play)] = data  
        
        self.item_list = list(self.df_cache.keys())
                            
    def _get_edge_index(self, n_players: int) -> TensorType["2", "num_edges"]: # TODO: Review docstring
        """ A method that computes the adjacency matrix of a fully connected graph with n_players nodes, 
            and later converts this matrix to a shape required by torch_geometric. 
                        
        Returns:
            TensorType["2", "num_edges"]: A tensor describing all the present connections between the nodes of the graph. 
            As the graph is modeled to be a fully connected graph, every node is connected with every other, except for itself. 
        """
        if n_players not in self.edge_index_cache: 
            A = (~torch.eye(n_players, dtype=torch.bool)).to(torch.int64) # TODO: Try out if torch.int8 also works 
            self.edge_index_cache[n_players] = A 
            
        return self.edge_index_cache[n_players]             
        
    def _build_data(self, df: pl.DataFrame, rows: int, file_type: str) -> Union[List[Data] | List[TensorType["*"]]]:
        n_frames =  df["frame_id"].n_unique()
        n_players = int(rows / n_frames) 
        n_features = 0
        groups = df.sort("frame_id").drop("frame_id").select(["x", "y", pl.exclude("x", "y")])  
        groups = groups.to_numpy().reshape(n_frames, n_players, -1)
        
        if file_type == "output": 
            groups = geometric_output_features(groups)   

        groups = torch.tensor(groups, dtype=torch.float32)
        
        if self.data_type == "graph": 
            A = self._get_edge_index(n_players=n_players)
            edge_index = torch.nonzero(torch.block_diag(*[A for _ in range(n_frames)])).T # this should already be done in the dataloader 
            x = groups.view(n_frames*n_players, -1)
            n_features = int(x.shape[-1]) 
            return Data(edge_index=edge_index, x=x), n_frames, n_players, n_features
             
        return groups
    
    def _normalize(self, df: pl.DataFrame, file_type: str) -> pl.DataFrame:
        """
            A method for normalizing, in the case of an input file, 
            every stated variable in the respective YAML file, and in the case of an output file, 
            the x-coordinates and the y-coordinates.
        
        Returns:
            pl.DataFrame: The pl.DataFrame with normalized columns.
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
            
        Returns:
            Dict[str, float]: A dictionary with the keys being the position name (str),
                              and the values being the embeddings associated with 
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
        if "input" in file:
            in_file = file
            out_file = file.replace("input", "output")
        else: 
            in_file = file.replace("output", "input")
            out_file = file
         
        if self.data_type == "graph": 
            source, n_frames_source, n_players_source, n_features_source = self.df_cache[(in_file, int(game), int(play))]
            if "train" in self.mode:
                target, n_frames_target, n_players_target, n_features_target = self.df_cache[(out_file, int(game), int(play))]  
            elif "predict" in self.mode: 
                target, n_frames_target, n_players_target, n_features_target = [None]*4  
        else: 
            source = self.df_cache[(in_file, int(game), int(play))] 
            if "train" in self.mode:
                target = self.df_cache[(out_file, int(game), int(play))]  
            elif "predict" in self.mode: 
                target = None
        
        data = {
                "source": source, 
                "target": target,
        }
        
        if self.data_type == "sequential":
            data = data | {
                "source_shape": source.shape[:2],
                "target_shape": target.shape[:2]
                }
            
            data = data | {
                "n_frames_source": n_frames_source, 
                "n_frames_target": n_frames_target,
                "n_features": n_features_source,  
                "n_players_source": n_players_source,
                "n_players_target": n_players_target, 
            }

        return data