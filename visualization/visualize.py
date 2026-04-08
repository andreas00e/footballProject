import os 
import math  
import hydra
import random
import logging 
import numpy as np 
import polars as pl 
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Union

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

OmegaConf.register_new_resolver("mult_2", lambda x, y: x * y)

logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)


class Visualize(): 
    def __init__(self,  i_play: pl.DataFrame, o_play: pl.DataFrame, bg_img: np.ndarray, ball_img: np.ndarray, ids: DictConfig, 
            features_load: DictConfig, features_norm: DictConfig, options: List, fig: DictConfig, extrema: DictConfig, animation: DictConfig) -> None:
        self.i_play = i_play
        self.o_play = o_play
        
        self.bg_img = bg_img
        self.ball_img = ball_img
        self.week_id, self.game_id, self.play_id = ids.values()
        
        self.features_load = features_load
        self.features_norm = features_norm
        
        self.options = options 
        self.x_size, self.y_size = list(fig.sizes.values())
        self.extrema = extrema
                
        self.animation = animation
            
        self.i_frames, self.o_frames = [0]*2
        self.id2info, self.play = {}, {}
        
        self._process_play()
        self.G = self._create_graph()
        
        self.fig, self.ax = plt.subplots(figsize=(self.x_size, self.y_size))
        self.text, self.anim = [None]*2 # later text and animation
             
    def _process_play(self) -> None:
        self.i_play[list(self.features_norm)] = self._normalize(self.i_play[list(self.features_norm)], self.extrema)
        self.o_play[["x", "y"]] = self._normalize(self.o_play[["x", "y"]], self.extrema)
        
        self.i_frames = int(self.i_play["frame_id"].n_unique())
        self.o_frames = int(self.o_play["frame_id"].n_unique())
        
        predictions, ids, _, names, _, sides = \
            self.i_play.filter(pl.col("frame_id") == 1)[["player_to_predict", "nfl_id", "play_direction", "player_name", "player_position", "player_side"]]
        
        names = names.map_elements(lambda x: x.split()[-1], return_dtype=pl.String)
        colors = sides.map_elements(lambda x: "red" if x == "Offense" else "blue")
        self.id2info = dict(zip(ids, zip(predictions, names, colors)))
                
        for id in ids: 
            if id in self.o_play["nfl_id"].unique(): # players that form part of both the input and the output
                self.play[id] = pl.concat([self.i_play.filter(pl.col("nfl_id") == id)[["x", "y"]], self.o_play.filter(pl.col("nfl_id") == id)[["x", "y"]]])
                self.play[id].insert_column(self.play[id].width, pl.Series("color", [self.id2info[id][-1]]*(self.i_frames+self.o_frames)))
                
            else: # players that form part of only the input 
                pad_element = pl.concat([self.i_play.filter(pl.col("nfl_id") == id)[["x", "y"]][-1]]*self.o_frames)
                self.play[id] = pl.concat([self.i_play.filter(pl.col("nfl_id") == id)[["x", "y"]], pad_element])
                self.play[id].insert_column(self.play[id].width, pl.Series("color", [self.id2info[id][-1] if i < self.i_frames else "grey" for i in range(self.i_frames+self.o_frames)]))

    def _create_graph(self) -> nx.Graph: 
        G = nx.Graph()
        G.add_nodes_from(self.play.keys())
        nx.set_node_attributes(G, values={}, name="pos")
        nx.set_node_attributes(G, values=[v[1] for v in self.id2info.values()], name="name")
        return G 
    
    def _update(self, frame_id) -> None:
        if frame_id: 
           self.ax.clear() 
           
        node_positions = {}
        node_color = []
        for k in self.play.keys(): 
            x, y, c = self.play[k][:frame_id+1]
            node_positions[k] = (x[-1], y[-1])
            node_color.append(c[-1])
            self.ax.plot(x, y, f"{c[-1][0]}--") # plot trace for player movements before ball is thrown 

        nx.draw_networkx_nodes(G=self.G, pos=node_positions, node_color=node_color, ax=self.ax)
        
        self.ax.set_title(f"Animation for play {self.play_id} from game {self.game_id} from week {self.week_id}")
        self.text = self.ax.text(0.9, 1.0, f"Frame:", transform=self.ax.transAxes, fontsize=12, color="black")
        self.text = self.ax.text(0.97, 1.0, f"{frame_id}", transform=self.ax.transAxes, fontsize=12, color="black")
        
        self.ax.imshow(self.bg_img, extent=[-1, 1, -1, 1], aspect="auto")

    def _normalize(self, df: pl.DataFrame, extrema: DictConfig) -> pl.DataFrame:
        return df.with_columns([
            (2*((pl.col(col) - extrema[col]["min"]) / (extrema[col]["max"] - extrema[col]["min"]))-1).alias(col)
            for col in df.columns
        ])

    def _ang2vec(self, input: float) -> Tuple[float, float]: 
        return math.sin(input), math.cos(input)
        
    def plot(self) -> None: 
        self.trace_lines = {n: self.ax.plot([], [], "r--")[0] for n in self.G.nodes} 
        i_traces = {node: [] for node in self.G.nodes}
        o_traces = {node: [] for node in self.G.nodes}
        
        self.i_frames = self.i_play["frame_id"].n_unique()
        self.o_frames = self.o_play["frame_id"].n_unique()
        n_frames = self.i_frames+self.o_frames
         
        plt.axis("equal")
        self.anim = animation.FuncAnimation(self.fig, self._update, frames=range(0, n_frames), repeat=False, blit=False)
        self.animation.fps *= n_frames
        self.anim.save(**self.animation)
   
def get_files(data_dir: os.PathLike, week: int, features: List[Union[int, str]]) -> Tuple[pl.DataFrame, pl.DataFrame]:  
    i_file = os.path.join(data_dir, "input_2023_w{:02d}.csv".format(week))
    i_df = pl.read_csv(i_file, columns=features)
    o_file = os.path.join(data_dir, "output_2023_w{:02d}.csv".format(week))
    o_df = pl.read_csv(o_file)    
    return i_df, o_df 

def get_frames(data_dir: os.PathLike, features: List[Union[int, str]], ids: Union[None, Dict]) -> Tuple[pl.Series, pl.Series]: 
    if ids:
        week, game_id, play_id = ids.values()
    else:  
        week = random.randint(a=1, b=18)
        
    i_df, o_df = get_files(data_dir=data_dir, week=week, features=features)
    
    if not ids: 
        game_id = random.choice(i_df["game_id"].unique())
        play_id = random.choice(i_df.filter(pl.col("game_id") == game_id)["play_id"].unique()) 
    
    print(f"Visualizing play {play_id} from game {game_id} from week {week}")
    
    i_play = i_df.filter((pl.col("game_id") == game_id) & (pl.col("play_id") == play_id))
    o_play = o_df.filter((pl.col("game_id") == game_id) & (pl.col("play_id") == play_id))
    return i_play, o_play


@hydra.main(config_path="../confs", config_name="visualize", version_base=None)
def main(cfg) -> None:
    OmegaConf.resolve(cfg)  
    
    bg_img = mpimg.imread(cfg.bg_img)
    ball_img = mpimg.imread(cfg.ball_img)
    data_dir = os.path.join(os.getcwd(), cfg.data_dir)
    features = cfg.features.load+cfg.features.norm
    
    if cfg.pick_rand: 
        ids = None 
    else: 
        ids=cfg.ids
        
    i_play, o_play = get_frames(data_dir=data_dir, features=features, ids=ids)

    visualize = Visualize(i_play=i_play, o_play=o_play, bg_img=bg_img, ball_img=ball_img, ids=ids,
                    features_load=cfg.features.load, features_norm=cfg.features.norm, **cfg.visualization)
    visualize.plot()
    
if __name__ == "__main__": 
    main()