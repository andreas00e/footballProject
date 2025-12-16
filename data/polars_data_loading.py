import os 
import numpy as np 
import polars as pl
from tqdm import tqdm 
from omegaconf import OmegaConf
from numpy.linalg import vector_norm as lavn
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
    x_x = np.pow(x, 2)
    y_y = np.pow(y, 2)
    
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
        np.ones_like(x) # bias
        ]
        
    return np.stack(features, axis=-1)
         
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
                data = self._build_data(df=frame, n_rows=frame.shape[0], data_type=self.data_type, file_type=file_type)
                self.df_cache[(file, game, play)] = data  
        
        self.item_list = list(self.df_cache.keys())
                            
    def _get_edge_index(self, n_players: int) -> TensorType["2", "num_edges"]:
        """ A method that computes the adjacency matrix of a fully connected graph with n_players nodes, 
            and later converts this matrix to a shape required by torch_geometric. 
                        
        Returns:
            TensorType["2", "num_edges"]: A tensor describing all the present connections between the nodes of the graph. 
            As the graph is modeled to be a fully connected graph, every node is connected with every other, except for itself. 
        """
        if n_players not in self.edge_index_cache:
            A = torch.ones((n_players, n_players), dtype=torch.float32) - torch.eye(n_players, dtype=torch.float32)
            self.edge_index_cache[n_players] = torch.stack(A.nonzero(as_tuple=True), dim=0)
        return self.edge_index_cache[n_players]                  
        
    def _build_data(self, df: pl.DataFrame, n_rows: int, data_type: str, file_type: str) -> Union[List[TensorType] | List[Data]]:
        n_frames =  df["frame_id"].n_unique()
        n_players = int(n_rows / n_frames) 
          
        if self.data_type == "graph": 
            edge_index = self._get_edge_index(n_players)
            
        groups = df.sort("frame_id").drop("frame_id")   
        if file_type == "input": 
            columns = groups.columns
            columns.remove("x")
            columns.remove("y")
            groups = groups.select(["x", "y"]+columns)
        
        groups = groups.to_numpy().reshape(n_frames, n_players, -1)
             
        if file_type == "output": 
            groups = geometric_output_features(groups)
         
        groups = torch.tensor(groups, dtype=torch.float32)
        return groups if data_type == "graph" else groups
    
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
             
        sources = self.df_cache[(in_file, int(game), int(play))] 
        targets = self.df_cache[(out_file, int(game), int(play))]            
        
        data = {
            "source": sources, 
            "target": targets,
            "source_shape": sources.shape[:2],
            "target_shape": targets.shape[:2]
            }
        
        return data