import os 
import math  
import hydra
import random
import logging 
import numpy as np 
import polars as pl 
from functools import partial
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

OmegaConf.register_new_resolver("mult_2", lambda x, y: x * y)

logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)


class Visualize(): 
    def __init__(self, identifiers: DictConfig, extrema: DictConfig, fig: DictConfig, in_play: pl.DataFrame, out_play: pl.DataFrame, _img: np.ndarray, info: Optional[bool]=False):
        self.week_id, self.game_id, self.play_id = identifiers.values()
        self.x_min, self.x_max, self.y_min, self.y_max = extrema.values()
        _, _, self.x_size, self.y_size = fig.values()
        
        self.in_play = in_play
        self.out_play = out_play
        self._img = _img
        self.info = info

        self.n_players, self.n_out_players, self.in_frames, self.out_frames = [0]*4
        self.players, self.out_players, self.x, self.y = [], [], [], []
        self.last_node_positions, self.trace_lines = {}, {}, {}
        
        self.players_dict = self._process_play()
        self.G = self._create_graph(self.players_dict)
        
        self.fig, self.ax = plt.subplots(figsize=(self.x_size, self.y_size))
        self.text, self.anim = None, None # later text, later animation
         
    def _process_play(self) -> Dict[str, str]:
        # TODO: Include line for play data that was predicted by the model
        self.n_players = self.in_play["nfl_id"].n_unique()
        self.out_players = [name.split()[1] for name in self.in_play.filter(pl.col("player_to_predict") == True)["player_name"].unique()]
        self.n_out_players = len(self.out_players)

        sides = self.in_play.filter(pl.col("frame_id") == 1)["player_side"]
        names = [name.split()[1] for name in self.in_play["player_name"].unique()]
        colors = ["red" if side == "Offense" else "blue" for side in sides]
        players_dict = dict(zip(names, colors))
        return players_dict
    
    def _create_graph(self, players_dict: Dict[str, str] ) -> nx.Graph: 
        G = nx.Graph()
        G.add_nodes_from(players_dict.keys())
        nx.set_node_attributes(G, values={}, name = "pos") # initialize node's position attribute
        nx.set_node_attributes(G, values=players_dict.keys(), name = "color") # XXX: Change color of nodes to reflect teams actual colors
        nx.set_node_attributes(G, values=players_dict.values(), name="name")
        return G 
    
    def _update(self, traces_in, traces_out, frame_id) -> None:
        if frame_id != 0: 
           self.ax.clear() 
        
        in_frames = self.in_frames
        if frame_id <= in_frames: # input 
            play = self.in_play.filter(pl.col("frame_id") == frame_id)
            color = "blue"
            # if self.info: 
            #     orientations = play["dir"] 
            #     orientations = list(map(self._ang2vec, orientations))
            #     node_orientations = dict(zip(self.players, orientations))
        else: # output
            play = self.out_play.filter(pl.col("frame_id") == frame_id-in_frames)
            color = "red"
        
        x_pos, y_pos = play["x", "y"]
        norm_x_pos = map(lambda x: self._normalize(x, self.x_min, self.x_max), x_pos)
        norm_y_pos = map(lambda y: self._normalize(y, self.y_min, self.y_max), y_pos)
        pos = list(zip(norm_x_pos, norm_y_pos))
        
        if frame_id <= in_frames: # input
            node_positions = dict(zip(self.players, pos)) # all players
            if frame_id == in_frames: 
                self.last_node_positions = node_positions # saving the final positions of all players (only appearing in the input sequence)
                # self.last_x_pos = x_pos
                # self.last_y_pos = y_pos
        elif frame_id > in_frames: # output
            node_positions = dict(zip(self.out_players, pos)) # only out players
            for player in self.players: 
                if player in self.out_players: 
                    self.last_node_positions[player] = node_positions[player]
                else: 
                    self.players_dict[player] = "grey"
            node_positions = self.last_node_positions
        
        # for node, attrs in self.G.nodes(data=True):  # draw direction vectors as arrows
        #     x, y = attrs.get("pos")

        #     traces_in[node].append((x, y)) # list of past input positions
        #     x_trace_in = [pos[0] for pos in traces_in[node]]
        #     y_trace_in = [pos[1] for pos in traces_in[node]]
        #     ax.plot(x_trace_in, y_trace_in, "r--") # plot trace for player movements before ball is thrown 
            
        #     if frame > len(n_input_frames): 
        #         traces_out[node].append((x, y)) # list of past input positions
        #         x_trace_out = [pos[0] for pos in traces_out[node]]
        #         y_trace_out = [pos[1] for pos in traces_out[node]]
        #         ax.plot(x_trace_out, y_trace_out, "b--") # plot trace for player movements when ball is in the air
        
        self.ax.set_title(f"Animation for play {self.play_id} from game {self.game_id} from week {self.week_id}")
        self.text = self.ax.text(0.9, 1.0, f"Frame:", transform=self.ax.transAxes, fontsize=12, color="black")
        self.text = self.ax.text(0.97, 1.0, f"{frame_id}", transform=self.ax.transAxes, fontsize=12, color=color)

        
        self.ax.imshow(self._img, extent=[-1, 1, -1, 1], aspect="auto")
        # ax.plot(x_pos, y_pos, "r--")
        # nx.set_node_attributes(self.G, node_positions, name="pos")
        # x, y = list(node_positions.values())[0]
        #self.x.append(x)
        # self.y.append(y)
        nx.draw_networkx_edges(G=self.G, pos=node_positions, ax=self.ax, edge_color="black")
        nx.draw_networkx_nodes(G=self.G, pos=node_positions, node_color=self.players_dict.values(), ax=self.ax)
        nx.draw_networkx_labels(G=self.G, pos=node_positions, labels=self.players, ax=self.ax, font_size=10, font_color="black")
        # ax.plot(self.x, self.y, "r--", lw=5)

    def _normalize(self, x: float, min: float, max: float) -> float:  
        return (x-min)/(max-min) 

    def _ang2vec(self, input: float) -> Tuple[float, float]: 
        return math.sin(input), math.cos(input)
        
    def plot(self) -> None: 
        self.trace_lines = {n: self.ax.plot([], [], "r--")[0] for n in self.G.nodes} 
        traces_in = {node: [] for node in self.G.nodes}
        traces_out = {node: [] for node in self.G.nodes}
        
        self.in_frames = self.in_play["frame_id"].n_unique()
        self.out_frames = self.out_play["frame_id"].n_unique()
        n_frames = self.in_frames+self.out_frames
        self.players = {k: k for k in self.players_dict}
 
        plt.axis("equal")
        self.anim = animation.FuncAnimation(self.fig, partial(self._update, traces_in, traces_out), frames=range(1, n_frames+1), repeat=False, blit=False)
        self.anim.save(
            filename="./visualization/play.mp4", 
            writer="ffmpeg", 
            fps=n_frames*0.25, 
            dpi=200.0, 
            extra_args=["-pix_fmt", "yuv420p"]
            ) 
   
def get_files(data_dir: os.PathLike, week: int, features: List[Union[int, str]]) -> Tuple[pl.DataFrame, pl.DataFrame]:  
    in_file = os.path.join(data_dir, "input_2023_w{:02d}.csv".format(week))
    in_df = pl.read_csv(in_file, columns=features)
    out_file = os.path.join(data_dir, "output_2023_w{:02d}.csv".format(week))
    out_df = pl.read_csv(out_file)    
    return in_df, out_df 

def get_frames(data_dir: os.PathLike, features: List[Union[int, str]], identifiers: Union[None, Dict]) -> Tuple[pl.Series, pl.Series]: 
    if identifiers:
        week, game_id, play_id = identifiers.values()
    else:  
        week = random.randint(a=1, b=18)
        
    in_df, out_df = get_files(data_dir=data_dir, week=week, features=features)
    
    if not identifiers: 
        game_id = random.choice(in_df["game_id"].unique().to_list())
        play_id = random.choice(in_df.filter(pl.col("game_id") == game_id)["play_id"].unique().to_list()) 
    
    print(f"Visualizing play {play_id} from game {game_id} from week {week}")
    
    in_play = in_df.filter((pl.col("game_id") == game_id) & (pl.col("play_id") == play_id))
    out_play = out_df.filter((pl.col("game_id") == game_id) & (pl.col("play_id") == play_id))
    return in_play, out_play


@hydra.main(config_path="../confs", config_name="visualize", version_base=None)
def main(cfg) -> None:  
    background_img = mpimg.imread(cfg.background_img)
    data_dir = os.path.join(os.getcwd(), cfg.data_dir)
    
    if cfg.pick_random: 
        identifiers = None 
    else: 
        identifiers=cfg.identifiers
    in_play, out_play = get_frames(data_dir=data_dir, features=cfg.features, identifiers=identifiers)

    visualize = Visualize(cfg.identifiers, cfg.extrema, cfg.fig, in_play, out_play, background_img)
    visualize.plot()
    
if __name__ == "__main__": 
    main()