import os 
import math  
import hydra
import numpy as np 
import pandas as pd
from functools import partial
from typing import Dict, Tuple

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.collections as mc
import matplotlib.animation as animation


class Visualize(): 
    def __init__(self, week: int, game_id: int, play_id: int, input_df: pd.DataFrame, output_df: pd.DataFrame, with_info: bool, background_image: np.ndarray):
        self.week = week 
        self.game_id = game_id
        self.play_id = play_id
        self.input_df = input_df
        self.output_df = output_df
        self.with_info = with_info
        self.background_image = background_image
        
        self.out_players = []
        self.final_dict, self.last_node_positions = {}, {}
        self.x, self.y = [], []
        
        self.x_min = 0.0
        self.x_max = 120.0
        self.y_min = 0.0
        self.y_max = 53.3
        
        self.input_play, self.output_play, self.player_dicts = self._load_play()
        self.G = self._create_graph(self.player_dicts)
        self.anim = None # later animation

    
    def _load_play(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        assert self.game_id in self.input_df['game_id'].unique(), ValueError('Stated game_id is not in the list of game_ids!')
        assert self.play_id in self.input_df['play_id'].unique(), ValueError('Stated play_id is not in the list of play_ids!')     

        input_play = self.input_df[(self.input_df['game_id'] == self.game_id) & (self.input_df['play_id'] == self.play_id)]
        output_play = self.output_df[(self.output_df['game_id'] == self.game_id) & (self.output_df['play_id'] == self.play_id)]  
                
        names = input_play['player_name'].unique() 
        self.out_players = input_play[input_play['player_to_predict'] == True]['player_name'].unique().tolist()
        sides = input_play[input_play['frame_id'] == 1]['player_side']
        colors = ['red' if side == 'Offense' else 'blue' for side in sides]
        player_dicts = dict(zip(names, colors))
        
        return input_play, output_play, player_dicts
    
    def _create_graph(self, player_dicts: Dict[str, str] ) -> nx.Graph: 
        G = nx.Graph()
        G.add_nodes_from(player_dicts.keys())
        nx.set_node_attributes(G, values = {}, name = 'pos') # initialize position attribute
        nx.set_node_attributes(G, values = player_dicts.keys(), name = 'color') # TODO: Change color of nodes to reflect teams actual colors
         
        return G 
    
    
    def _update(self, ax, traces_in, traces_out, frame) -> None:  
        assert 0 < frame <= sum(self.final_dict.values()), ValueError('Stated frame is not included in frame indices')
        n_input_frames = self.final_dict['input_frames']
        players = list(self.player_dicts.keys()) # all player # TODO: Move to __init__() to avoid unnecessary repetition 
        
        if frame <= n_input_frames:
            play = self.input_play[self.input_play['frame_id'] == frame]
            if self.with_info: 
                orientations = play['dir'] 
                orientations = list(map(self._ang2vec, orientations))
                node_orientations = dict(zip(players, orientations))
        else:  
            play = self.output_play[self.output_play['frame_id'] == frame-n_input_frames]
        
        x_pos = [x for x in play['x']]
        y_pos = [y for y in play['y']]
        norm_x_pos = map(lambda x: self._normalize(x, self.x_min, self.x_max), x_pos)
        norm_y_pos = map(lambda y: self._normalize(y, self.y_min, self.y_max), y_pos)
        
        positions = [pos for pos in zip(norm_x_pos, norm_y_pos)]
        
        if frame <= n_input_frames: 
            node_positions = dict(zip(players, positions)) 
            if frame == n_input_frames: 
                self.last_node_positions = node_positions # Save last node positions of input sequence to continue with them for the output frames
                # self.last_x_pos = x_pos
                # self.last_y_pos = y_pos
        elif frame > n_input_frames: 
            node_positions = dict(zip(self.out_players, positions))
            for player in players: 
                if player in self.out_players: 
                    self.last_node_positions[player] = node_positions[player]
                else: 
                    self.player_dicts[player] = 'grey'
            node_positions = self.last_node_positions
        
        # for node, attrs in self.G.nodes(data=True):  # draw direction vectors as arrows
        #     x, y = attrs.get('pos')

        #     traces_in[node].append((x, y)) # list of past input positions
        #     x_trace_in = [pos[0] for pos in traces_in[node]]
        #     y_trace_in = [pos[1] for pos in traces_in[node]]
        #     ax.plot(x_trace_in, y_trace_in, 'r--') # plot trace for player movements before ball is thrown 
            
        #     if frame > len(n_input_frames): 
        #         traces_out[node].append((x, y)) # list of past input positions
        #         x_trace_out = [pos[0] for pos in traces_out[node]]
        #         y_trace_out = [pos[1] for pos in traces_out[node]]
        #         ax.plot(x_trace_out, y_trace_out, 'b--') # plot trace for player movements when ball is in the air
        
        ax.clear()
        ax.imshow(self.background_image, extent=[-1, 1, -1, 1], aspect='auto')
        # ax.plot(x_pos, y_pos, 'r--')
        # nx.set_node_attributes(self.G, node_positions, name='pos')
        x, y = list(node_positions.values())[0]
        self.x.append(x)
        self.y.append(y)
        nx.draw(self.G, ax=ax, with_labels=True, node_color=self.player_dicts.values(), pos=node_positions) # draw danymic graph
        ax.plot(self.x, self.y, 'r--', lw=5)

    def _normalize(self, input: float, min: float, max: float) -> float: 
        return 2 * (input - min) / (max - min) - 1

    def _ang2vec(self, input: float) -> Tuple[float, float]: 
        return math.sin(input), math.cos(input)
    
    def _init(): 
        pass
    
    
    def plot(self) -> None: 
        fig, ax = plt.subplots()
        self.trace_lines = {n: ax.plot([], [], 'r--')[0] for n in self.G.nodes} 
        traces_in = {node: [] for node in self.G.nodes}
        traces_out = {node: [] for node in self.G.nodes}
        
        input_frames = self.input_play['frame_id'].unique()
        output_frames = self.output_play['frame_id'].unique()
        self.final_dict = {'input_frames': input_frames.size, 'output_frames': output_frames.size}
        n_frames = sum(self.final_dict.values())
 
        plt.axis('equal')
        self.anim = animation.FuncAnimation(fig, partial(self._update, ax, traces_in, traces_out), frames=range(1, n_frames+1), repeat=False, blit=False)
        self.anim.save('growingCoil.mp4', writer='ffmpeg', fps=n_frames*0.25) 
    
@hydra.main(config_path='../conf', config_name='visualize', version_base=None)
def main(cfg) -> None: 
    data_dir = os.path.join(os.getcwd(), cfg.data_dir)
    background_image = mpimg.imread(cfg.background_image)
        
    week = cfg.week
    assert week in list(range(1, 19)), ValueError('We only have data for week 1 to week 18 from the 2023 NFL regular season. \n \
        The stated week is not in the dataset!')
    
    input_file = os.path.join(data_dir, 'input_2023_w{:02d}.csv'.format(week))
    input_df = pd.read_csv(input_file, index_col=False, usecols=cfg.feature_config)
    
    output_file = os.path.join(data_dir, 'output_2023_w{:02d}.csv'.format(week))
    output_df = pd.read_csv(output_file, index_col=False)
    
    visualize = Visualize(week, cfg.game_id, cfg.play_id, input_df, output_df, False, background_image)
    visualize.plot()
    
    
if __name__ == '__main__': 
    main()