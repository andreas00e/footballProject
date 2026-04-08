import os 
import math  
import hydra
import random
import logging 
import numpy as np 
import polars as pl 
from functools import partial
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Union

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

OmegaConf.register_new_resolver("mult_2", lambda x, y: x * y)

logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)


class Visualize(): 
    def __init__(self,  i_play: pl.DataFrame, o_play: pl.DataFrame, bg_img: np.ndarray, ids: DictConfig, features_load: DictConfig, 
                features_norm: DictConfig, options: List, fig: DictConfig, extrema: DictConfig, animation: DictConfig) -> None:
        self.i_play = i_play
        self.o_play = o_play
        
        self.bg_img = bg_img
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
        nx.set_node_attributes(G, values=self.id2name, name="name")
        return G 
    
    def _update(self, traces_in, traces_out, frame_id) -> None:
        if frame_id: 
           self.ax.clear() 
           
        node_positions = {}
        for k in self.play.keys(): 
            node_positions[k] = self.play[k][frame_id].row(0)
        
        nx.draw_networkx_nodes(G=self.G, pos=node_positions, ax=self.ax, node_color=self.id2color.values())
        

        # in_frames = self.i_frames
        # if frame_id <= in_frames: # input part of the sequence  
        #     play = self.i_play.filter(pl.col("frame_id") == frame_id)
        #     # if self.orientation: 
        #     #     orientations = play["dir"] 
        #     #     orientations = list(map(self._ang2vec, orientations))
        #     #     node_orientations = dict(zip(self.players, orientations))
        # else: # output part of the sequence 
        #     play = self.o_play.filter(pl.col("frame_id") == frame_id-in_frames)
        
        # x_pos, y_pos = play["x", "y"]
        # pos = list(zip(x_pos, y_pos))
        
        # if frame_id <= in_frames: # input
        #     node_positions = dict(zip(self.players.keys(), pos)) # all players
        #     if frame_id == in_frames: 
        #         self.last_node_positions = node_positions # saving the final positions of all players (only appearing in the input sequence)
        #         # self.last_x_pos = x_pos
        #         # self.last_y_pos = y_pos
        # elif frame_id > in_frames: # output
        #     node_positions = dict(zip(self.out_players, pos)) # only out players
        #     for player in self.players.keys(): 
        #         if player in self.out_players: 
        #             self.last_node_positions[player] = node_positions[player]
        #         else: 
        #             self.players[player] = *self.players[player][:2], "grey"
        #     node_positions = self.last_node_positions
            

        
        # # for node, attrs in self.G.nodes(data=True):  # draw direction vectors as arrows
        # #     x, y = attrs.get("pos")
        # #     traces_in[node].append((x, y)) # list of past input positions
        # #     x_trace_in = [pos[0] for pos in traces_in[node]]
        # #     y_trace_in = [pos[1] for pos in traces_in[node]]
        # #     ax.plot(x_trace_in, y_trace_in, "r--") # plot trace for player movements before ball is thrown 
            
        # #     if frame > len(n_input_frames): 
        # #         traces_out[node].append((x, y)) # list of past input positions
        # #         x_trace_out = [pos[0] for pos in traces_out[node]]
        # #         y_trace_out = [pos[1] for pos in traces_out[node]]
        # #         ax.plot(x_trace_out, y_trace_out, "b--") # plot trace for player movements when ball is in the air
        
        # self.ax.set_title(f"Animation for play {self.play_id} from game {self.game_id} from week {self.week_id}")
        # self.text = self.ax.text(0.9, 1.0, f"Frame:", transform=self.ax.transAxes, fontsize=12, color="black")
        # # self.text = self.ax.text(0.97, 1.0, f"{frame_id}", transform=self.ax.transAxes, fontsize=12, color=color)

        
        self.ax.imshow(self.bg_img, extent=[-1, 1, -1, 1], aspect="auto")
        # # ax.plot(x_pos, y_pos, "r--")
        # # nx.set_node_attributes(self.G, node_positions, name="pos")
        # # x, y = list(node_positions.values())[0]
        # #self.x.append(x)
        # # self.y.append(y)
        
        # nx.draw_networkx_edges(G=self.G, pos=node_positions, ax=self.ax, edge_color="black")
        # nx.draw_networkx_nodes(G=self.G, pos=node_positions, node_color=[v[-1] for v in self.players.values()], ax=self.ax)

        # # nx.draw_networkx_nodes(G=self.G, pos=node_positions, node_color=[c for _, _, c in self.players.values()], ax=self.ax)

        # # shapes = list(set(value[0] for value in self.players.values()))
        
        # # for k, v in self.players.items(): 
        # #     self.groups[v[0]][k] = v
        
        # # for k_1, v_1 in self.groups.items(): 
        # #     for k_2, v_2 in v_1.items(): 
        # #         pos = node_positions

        # #     nx.draw_networkx_nodes(G=self.G, pos=node_positions, node_color=[c for _, _, c in self.players.values()], ax=self.ax)
        # # nx.draw_networkx_labels(G=self.G, pos=node_positions, labels=self.players.keys(), ax=self.ax, font_size=10, font_color="black")
        # # ax.plot(self.x, self.y, "r--", lw=5)

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
        self.anim = animation.FuncAnimation(self.fig, partial(self._update, i_traces, o_traces), frames=range(0, n_frames), repeat=False, blit=False)
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
    data_dir = os.path.join(os.getcwd(), cfg.data_dir)
    features = cfg.features.load+cfg.features.norm
    
    if cfg.pick_rand: 
        ids = None 
    else: 
        ids=cfg.ids
        
    i_play, o_play = get_frames(data_dir=data_dir, features=features, ids=ids)

    visualize = Visualize(i_play=i_play, o_play=o_play, bg_img=bg_img, ids=ids,
                    features_load=cfg.features.load, features_norm=cfg.features.norm, **cfg.visualization)
    visualize.plot()
    
if __name__ == "__main__": 
    main()