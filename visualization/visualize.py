import os 
import math  
import hydra
import numpy as np 
import polars as pl 
from functools import partial
from typing import Dict, Optional, Tuple

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.collections as mc
import matplotlib.animation as animation


class Visualize(): 
    def __init__(self, identifiers: Dict, extrema: Dict, in_df: pl.DataFrame, out_df: pl.DataFrame, background_img: np.ndarray, info: Optional[bool]=False):
        self.week, self.game_id, self.play_id = identifiers.values()
        self.x_min, self.x_max, self.y_min, self.y_max = extrema.values()
        self.in_df = in_df
        self.out_df = out_df
        self.background_img = background_img
        self.info = info

        self.n_players, self.n_out_players = 0, 0
        self.players, self.out_players, self.x, self.y = [], [], [], []
        self.final_dict, self.last_node_positions = {}, {}
        
        self.in_play, self.out_play, self.player_dicts = self._load_play()
        self.G = self._create_graph(self.player_dicts)
        self.anim = None # later animation
        
    
    def _load_play(self) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, str]]:
        assert self.game_id in self.in_df["game_id"].unique().to_list(), ValueError("Stated game_id is not in the list of game_ids")
        assert self.play_id in self.in_df["play_id"].unique().to_list(), ValueError("Stated play_id is not in the list of play_ids")     
        
        in_play = self.in_df.filter((pl.col("game_id") == self.game_id) & (pl.col("play_id") == self.play_id))
        out_play = self.out_df.filter((pl.col("game_id") == self.game_id) & (pl.col("play_id") == self.play_id))
        # TODO: Include line for play data that was predicted by the model
        
        self.n_players = len(in_play["nfl_id"].unique())
        self.out_players = [name.split()[1] for name in in_play.filter(pl.col("player_to_predict") == True)["player_name"].unique().to_list()]
        self.n_out_players = len(self.out_players)

        sides = in_play.filter(pl.col("frame_id") == 1)["player_side"]
        names = [name.split()[1] for name in in_play["player_name"].unique()]
        colors = ["red" if side == "Offense" else "blue" for side in sides]
        player_dict = dict(zip(names, colors))
        return in_play, out_play, player_dict
    
    def _create_graph(self, player_dicts: Dict[str, str] ) -> nx.Graph: 
        G = nx.Graph()
        G.add_nodes_from(player_dicts.keys())
        nx.set_node_attributes(G, values = {}, name = "pos") # initialize nodes/players position attribute
        nx.set_node_attributes(G, values = player_dicts.keys(), name = "color") # TODO: Change color of nodes to reflect teams actual colors
        return G 
    
    def _update(self, ax, traces_in, traces_out, frame_id) -> None:  
        assert 0 < frame_id <= self.n_frames, ValueError("Stated frame is not included in frame indices")
        n_input_frames = self.final_dict["in_frames"] # TODO: Move this to __init__
        
        if frame_id <= n_input_frames: # input 
            play = self.in_play.filter(pl.col("frame_id") == frame_id)
            # if self.info: 
            #     orientations = play["dir"] 
            #     orientations = list(map(self._ang2vec, orientations))
            #     node_orientations = dict(zip(self.players, orientations))
        else: # output
            play = self.out_play.filter(pl.col("frame_id") == frame_id-n_input_frames)
        
        x_pos, y_pos = play["x", "y"]
        norm_x_pos = map(lambda x: self._normalize(x, self.x_min, self.x_max, mode="vanilla"), x_pos) # TODO: Move mode to config 
        norm_y_pos = map(lambda y: self._normalize(y, self.y_min, self.y_max, mode="vanilla"), y_pos)
        pos = list(zip(norm_x_pos, norm_y_pos))
        
        if frame_id <= n_input_frames: # input
            node_positions = dict(zip(self.players, pos)) # all players
            if frame_id == n_input_frames: 
                self.last_node_positions = node_positions # saving the final positions of all players (only appearing in the input sequence)
                # self.last_x_pos = x_pos
                # self.last_y_pos = y_pos
        elif frame_id > n_input_frames: # output
            node_positions = dict(zip(self.out_players, pos)) # only out players
            for player in self.players: 
                if player in self.out_players: 
                    self.last_node_positions[player] = node_positions[player]
                else: 
                    self.player_dicts[player] = "grey"
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
        
        ax.clear()
        ax.imshow(self.background_img, extent=[-1, 1, -1, 1], aspect="auto")
        # ax.plot(x_pos, y_pos, "r--")
        # nx.set_node_attributes(self.G, node_positions, name="pos")
        # x, y = list(node_positions.values())[0]
        #self.x.append(x)
        # self.y.append(y)
        nx.draw(self.G, ax=ax, with_labels=True, node_color=self.player_dicts.values(), pos=node_positions)
        # ax.plot(self.x, self.y, "r--", lw=5)

    def _normalize(self, x: float, min: float, max: float, mode:str) -> float:  
        if mode == "vanilla": 
            return (x-min)/(max-min) 
        elif mode == "symmetric": 
            return 2*(x-min)/(max-min)-1

    def _ang2vec(self, input: float) -> Tuple[float, float]: 
        return math.sin(input), math.cos(input)
        
    def plot(self) -> None: 
        fig, ax = plt.subplots(figsize=(12, 8))
        self.trace_lines = {n: ax.plot([], [], "r--")[0] for n in self.G.nodes} 
        traces_in = {node: [] for node in self.G.nodes}
        traces_out = {node: [] for node in self.G.nodes}
        
        self.final_dict = {"in_frames": len(self.in_play["frame_id"].unique()), "out_frames": len(self.out_play["frame_id"].unique())}
        self.n_frames = sum(self.final_dict.values())
        self.players = list(self.player_dicts.keys())
 
        plt.axis("equal")
        self.anim = animation.FuncAnimation(fig, partial(self._update, ax, traces_in, traces_out), frames=range(1, self.n_frames+1), repeat=False, blit=False)
        self.anim.save(
            filename="growingCoil.mp4", 
            writer="ffmpeg", 
            fps=self.n_frames*0.25, 
            dpi=200.0, 
            extra_args=["-pix_fmt", "yuv420p"]
            ) 
    
@hydra.main(config_path="../confs", config_name="visualize", version_base=None)
def main(cfg) -> None:  
    background_img = mpimg.imread(cfg.background_img)
    data_dir = os.path.join(os.getcwd(), cfg.data_dir)
    week = cfg.identifiers.week
    assert week in list(range(1, 19)), ValueError("Available data ranges from week 1 to week 18 of the 2023 NFL regular season. \n \
        The stated week is not in the dataset!")
    
    in_file = os.path.join(data_dir, "input_2023_w{:02d}.csv".format(week))
    in_df = pl.read_csv(in_file, columns=cfg.features)
    out_file = os.path.join(data_dir, "output_2023_w{:02d}.csv".format(week))
    out_df = pl.read_csv(out_file)
    
    visualize = Visualize(cfg.identifiers, cfg.extrema, in_df, out_df, background_img)
    visualize.plot()
    

if __name__ == "__main__": 
    main()