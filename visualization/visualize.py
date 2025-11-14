import os 
import pandas as pd

import math  

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

X_MIN, X_MAX = 0.0, 120.0
Y_MIN, Y_MAX = 0.0, 53.3

G = nx.Graph() 
fig, ax = plt.subplots()

week = 5

assert week in list(range(1, 19)), 'Week is not part of weeks' # TODO: Think of a better assertion error message!

input_path = '/home/ehre/Documents/footballProject/data/nfl-big-data-bowl-2026-prediction/train/input_2023_w{:02d}.csv'.format(week)
output_path = '/home/ehre/Documents/footballProject/data/nfl-big-data-bowl-2026-prediction/train/output_2023_w{:02d}.csv'.format(week)

background_path = '/home/ehre/Documents/footballProject/data/images/field.jpeg'
background = mpimg.imread(background_path)
pos = nx.spring_layout(G)
    
if os.path.isfile(input_path) and os.path.isfile(output_path): 
    
    input_data = pd.read_csv(input_path) # open input file (input training data)
    output_data = pd.read_csv(output_path) # open output file (output training data)
    
    game_ids = input_data['game_id'].unique().tolist() # all games played during the week that is currently looked at 
    game_id = 0 # TODO: create config file in which users can state the game they want to inspect
    assert -1 < game_id < len(game_ids), 'Stated game_id does not match any of the given game ids'
    game_id = game_ids[game_id] # game we want to look at 
    
    play_ids = input_data[input_data['game_id'] == game_id]['play_id'].unique().tolist() # all plays of the current game
    play_id = 0 # TODO: create config file in which users can state the play they want to inspect
    assert -1 < play_id < len(play_ids), 'Stated game_id does not match any of the given game ids'
    play_id = play_ids[play_id] # game we want to look at 
    
    input_play = input_data[(input_data['game_id'] == game_id) & (input_data['play_id'] == play_id)]
    output_play = output_data[(output_data['game_id'] == game_id) & (output_data['play_id'] == play_id)]

    columns = input_play.columns.to_list() # names of every column
    columns_of_interest = ['nfl_id','frame_id','player_name','player_side','x','y','s','a','dir'] # banes of columns we are interested in 
    assert all(i in columns for  i in columns_of_interest), 'At least one column of interest is not present in the DataFrame'
    
    frame_1 = input_play[input_play['frame_id'] == 1][columns_of_interest] # get first frame of current game and play

    players = frame_1['player_name'].tolist() # all present on-field players of current play 
    ids = {player: player_id for player, player_id in zip(players, frame_1['nfl_id'].unique().tolist())}
    player_side = frame_1['player_side'].tolist()
    
    colors = ['red' if side == 'Offense' else 'blue' for side in player_side]
    player_colors = dict(zip(players, colors))
   
    G.add_nodes_from(players) # add nodes (on-field players of current play) to graph
    # nx.set_node_attributes(G=G, values=player_colors, name='color') # set node colors TODO: Change color of nodes to reflect teams actual colors
    
    trace_lines = {n: ax.plot([], [], 'r--')[0] for n in G.nodes} # trace lines 
    traces_in = {node: [] for node in G.nodes}
    traces_out = {node: [] for node in G.nodes}

    input_frames = input_play['frame_id'].unique().tolist()
    output_frames = output_data[(output_data['game_id'] == game_id) & ((output_data['play_id'] == play_id))]['frame_id'].unique().tolist()
    output_frames = [f+len(input_frames) for f in output_frames]
    
    input_dict = dict(zip(input_frames, ['i' for _ in range(len(input_frames))]))
    output_dict = dict(zip(output_frames, ['0' for _ in range(len(output_frames))]))

    final_dict = input_dict | output_dict
    
    last_node_positions = {}
    
def normalize(input:float, min:float, max:float): 
    return 2*(input-min)/(max-min)-1

def ang2vex(input:float): 
    return math.sin(input), math.cos(input)
    
def update(frame):  
    global last_node_positions      
    assert 0 < frame < len(final_dict)+1, "Specified frame is not part of frame indices"
    if frame <= len(input_dict):
        plays = input_play[input_play['frame_id'] == frame]
        # orientations = plays['dir'] 
        # orientations = list(map(ang2vex, orientations))
        # node_orientations = dict(zip(players, orientations))
    else:  
        plays = output_play[output_play['frame_id'] == frame-len(input_dict)]
    
    positions = plays[['x', 'y']]
    x_positions = [row.x for row in positions.itertuples(index=False)]
    y_positions = [row.y for row in positions.itertuples(index=False)]
    x_positions = map(lambda x: normalize(x, X_MIN, X_MAX), x_positions)
    y_positions = map(lambda y: normalize(y, Y_MIN, Y_MAX), y_positions)
    
    positions = [pos for pos in zip(x_positions, y_positions)]
      
    if frame <= len(input_dict): 
        node_players = players
        node_positions = dict(zip(node_players, positions))
        if frame == len(input_dict): 
            last_node_positions = node_positions
    else: 
        out_players = [name for name in ids.keys() if ids[name] in output_play['nfl_id'].unique().tolist()]
        in_players = [name for name in players if name not in out_players]
        
        for i, out_player in enumerate(out_players): 
            last_node_positions[out_player] = positions[i]
            node_positions = last_node_positions
        
        for in_player in in_players:
            colors[list(last_node_positions).index(in_player)] = 'grey'
        
    # draw danymic graph
    ax.clear()
    ax.imshow(background, extent=[-1, 1, -1, 1], aspect='auto')
    nx.set_node_attributes(G, node_positions, name='pos')
    # nx.set_node_attributes(G, node_orientations, name='orientations')
    nx.draw(G, ax=ax, with_labels=True, node_color=colors, pos=node_positions)
    
    # draw direction vectors as arrows
    for node, attrs in G.nodes(data=True): 
        x, y = attrs.get('pos')

        traces_in[node].append((x, y)) # list of past input positions
        x_trace_in = [pos[0] for pos in traces_in[node]]
        y_trace_in = [pos[1] for pos in traces_in[node]]
        ax.plot(x_trace_in, y_trace_in, 'r--') # plot trace for player movements before ball is thrown 
        
        if frame > len(input_dict): 
            traces_out[node].append((x, y)) # list of past input positions
            x_trace_out = [pos[0] for pos in traces_out[node]]
            y_trace_out = [pos[1] for pos in traces_out[node]]
            ax.plot(x_trace_out, y_trace_out, 'b--') # plot trace for player movements when ball is in the air

            

def main():  
    ani = animation.FuncAnimation(fig, update, frames=list(range(1, len(final_dict)+1)), repeat=False, interval=100)
    plt.axis('equal')
    plt.show()
        

if __name__ == '__main__': 
    main() 