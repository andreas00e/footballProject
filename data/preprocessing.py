import os 
import math 
import argparse
import numpy as np 
import pandas as pd
import polars as pl 
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf as oc
from typing import Dict, List, Tuple, Union

# QB: Quarterback       # DG: Defensive Guard       # K: Kicker 
# RB: Runningback       # DT: Defensive Tackle      # P Punter
# FB: Fullback          # NT: Nose Tackle 
# WR: Wide Reciever     # DE: Defensive End 
# TE: Tight End         # LB: Linebacker 
# C: Center             # ILB: Inside Linebacker 
# T: Tackle             # OLB: Outside Linebacker 
# OT: Offensive Tackle  # MLB: Middle Linebacker 
# OG: Offensive Guard   # CB: Corner Back 
                        # S: Safety 
                        # FS: Free Safety 
                        # SS: Strong Safety 
                        
# For more information about the positions of american football https://en.wikipedia.org/wiki/American_football_positions

N_POSITIONS = {
 "QB": 0.0, "RB": 0.0, "FB": 0.0, "WR": 0.0, "TE": 0.0, "C": 0.0, "T": 0.0, "OT": 0.0, "OG": 0.0,
 "DG": 0.0, "DT": 0.0, "NT": 0.0, "DE": 0.0, "LB": 0.0, "ILB": 0.0, "OLB": 0.0, "MLB": 0.0,
 "CB": 0.0, "S": 0.0, "FS": 0.0, "SS": 0.0, "K": 0.0, "P": 0.0
}

def get_pos(csv_path: os.PathLike) -> Union[List[str], None]:
    """ Method returning the name of every position players have in the currently looked at csv file 

    Args:
        csv_path (os.PathLike): Path to the csv file containing the play data

    Returns:
        Union[List[str], None]: List of all unique positions, if the given csv file was an input file.
                                If the given csv file was an output file, None is returned 
    """
    
    if "input" in csv_path:
        pos = []
        df = pd.read_csv(csv_path)
        pos = df["player_position"].unique().tolist()
        return pos 
    else: 
        return None

def pos2num(file_path: os.PathLike) -> Dict[str: float]: 
    """ Method that transforms literal position labels into numeric position values that can later be passed to the NN
    
    Args:
        file_path (os.PathLike): Path to the text file containing every positions literal, e.g. QB 

    Returns:
        dict[str: float]: Dictionary with the keys being the position literals, and the keys being the encoded numeric position values 
    """
    with open(file_path) as f: 
        pos_lit = [pos.replace(" ", "") for pos in f.read().split("\n")] # literal of position
        pos_num = list(map(lambda x: math.sin(math.sqrt(int.from_bytes(x.encode('utf-8'), 'big'))), pos_lit)) # value of position 
        return dict(zip(pos_lit, pos_num))
    
def get_max_positions(file_path: os.PathLike) -> Tuple[int, int]: 
    n_in_players, n_out_players =  0, 0
    
    df = pl.read_csv(source=file_path, columns=["game_id", "play_id", "nfl_id"])
    
    games = df["game_id"].unique()
    plays = df["play_id"].unique() 
    
    for game in games: 
        for play in plays: 
            n_players = df[(df['game_id'] == game) & (df['play_id'] == play)]['nfl_id'].nunique() 
            if 'input' in file_path: 
                n_in_players = n_players
            else: 
                week = file_path.split('/')[-1]
                if n_players > 7: 
                    print(f"{week} and {game} and {play}")
                    print(n_players)
                n_out_players = n_players
                
    
    return n_in_players, n_out_players




            
            
    
    # for _, df_i in df.groupby(['game_id', 'play_id']): 
    #     if 'input' in file_path: 
    #         in_players = df_i['nfl_id'].nunique()
    #         out_players = 0
    #     if 'output' in file_path: 
    #         print(file_path)
    #         in_players = 0 
    #         out_players = df_i['nfl_id'].nunique()
    #         print(out_players)
            
    #     return in_players, out_players

def get_max_frames(file_path: os.PathLike) -> Tuple[int, int]:      
    df = pd.read_csv(file_path)
    
    
    if 'input' in file_path: 
        return df['frame_id'].max(), 0
    elif 'output' in file_path: 
        return 0, df['frame_id'].max()
    else: 
        return None 
    
def get_n_positions(file_path: os.PathLike) -> Dict:
    df = pd.read_csv(file_path)
    
    for _, df_i in df.groupby(['game_id', 'play_id']): 
        player_counts = dict(df_i[(df_i['frame_id']==1) & (df_i['player_to_predict']==True)]['player_position'].value_counts())
        for key in player_counts.keys(): 
            if player_counts[key] > N_POSITIONS[key]: 
                 N_POSITIONS[key] = player_counts[key]
        
    return N_POSITIONS

def get_min_max(file: Path, features: List[str]): # min_max_conf is a DictConfig...
    data = pd.read_csv(file)
    if 'player_height' in data.columns.name: # input file 
        data['player_height'] = data['player_height'].apply(lambda x: (int(x.split('-')[0])*12.0+int(x.split('-')[1]))*2.54)
        min_max_dict = dict(zip(features, zip(data[features].min(), data[features].max())))

    else: # output file
        min_max_dict = dict(zip(features, zip((np.inf, -np.inf) for _ in range(len(features)))))
        features = ['x', 'y']
        min_max_dict['x'] = data[features].min()
        min_max_dict['y'] = data[features].max() 
    return min_max_dict

def transform(files: List[Path], flip: str ='both') -> None: 
    p_bar = tqdm(files, colour='green')
    for file in p_bar: 
        p_bar.set_description("Processing: {}".format(os.path.basename(file)))
        data = pd.read_csv(file)
        file_path, file_name = os.path.split(file)
        parent_path = os.path.dirname(file_path)
        
        if not 'train_inv' in os.listdir(parent_path):  # check if folder containing invariant csv files already exists
            os.mkdir(os.path.join(parent_path, 'train_inv')) # create folder for holding invariant data
            
        file_path_inv = os.path.join(parent_path, 'train_inv')
        file_name_inv = file_name.split('.')[0]+'_inv.csv' # file name for csv file containing holding plays   
        file_inv = os.path.join(file_path_inv, file_name_inv)
              
        data_inv = {}   
        data_inv['game_id'] = data['game_id']
        data_inv['play_id'] = data['play_id']
        data_inv['nfl_id'] = data['nfl_id']
        match flip: # TODO: Make match case statement less redundant
            case '⇅': # flip along x-axis ( → ), change of player positioning 
                data_inv['x_⇅'] = list(120.0 - data['x'])
                if 'input' in file: 
                    data_inv['o_⇅'] = list(180.0 - data['o'])
                    data_inv['dir_⇅'] = list(180.0 - data['dir'])

            case '⇄': # flip along y-axis ( ↑ ), change of play direction 
                data_inv['y_⇄'] = list(53.3 - data['y'])
                if 'input' in file: 
                    data_inv['o_⇄'] = list(360.0 - data['o'])
                    data_inv['dir_⇄'] = list(360.0 - data['dir'])
                    data_inv['play_direction_⇄'] = list(1.0 - data['play_direction'].apply(lambda x: 0.0 if x == 'left' else 1.0))
                
            case 'both': # flip along both x-axis ( → ) and y-axis ( ↑ ), change of both player positioning and play direction 
                data_inv['x_inv'] = list(120.0 - data['x'])
                data_inv['y_inv'] = list(53.3 - data['y'])
                if 'input' in file: 
                    data_inv['o_inv'] = list((data['o'] + 180.0) % 360.0)
                    data_inv['dir_inv'] = list((data['dir']+180.0) % 360.0)
                    data_inv['play_direction_inv'] = list(1.0 - data['play_direction'].apply(lambda x: 0.0 if x == 'left' else 1.0))
                    
        data_inv = pd.DataFrame.from_dict(data=data_inv)
        data_inv = data_inv.to_csv(file_inv, index=False)
        
        
def main(): 
    global N_POSITIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='conf/train.yaml', help='Provide the path to the configuartion file needed for training.')
    parser.add_argument('--pos_path', default='data/positions/positions.txt', help='Provide the path to the text file storing all positions.')
    parser.add_argument('--min_max_path', default='conf/min_max_values.yaml', help='Provide the path to the configuartion file needed for normalization.')
    args = parser.parse_args()
    
    train_path = os.path.join(os.getcwd(), args.train_path) # TODO: Change name of yaml file so that it does not seem as if it was only used for training the model (or use multiple yaml files)
    train_conf = oc.load(train_path) 
    min_max_path = os.path.join(os.getcwd(), args.min_max_path)
    min_max_conf = oc.load(min_max_path)
    
    # output files don't contain any variables we don't already know the min and max values of
    data_path = train_conf['data']['dataset']['data_dir']
    # files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv') and 'input' in file]
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv')]
    features = train_conf['feature_config']['model']['norm']
    
    # infile = None
    # out_file = None
    # in_max_frames, out_max_frames = 0, 0 
    # for file in tqdm(files, colour='green'): 
    #     in_frames, out_frames = get_max_frames(file)
    #     if in_frames > in_max_frames: 
    #         in_max_frames = in_frames
    #         in_file = file
    #     if out_frames > out_max_frames:
    #         out_max_frames = out_frames
    #         out_file = file
    
    # print("Maximum number of frames in an input play for all week, games, and plays: {n}, {f}".format(n=in_max_frames, f=in_file))
    # print("Maximum number of frames in an output play for all week, games, and plays: {n}, {f}".format(n=out_max_frames, f=out_file))

            
    # exit()
        

    # in_max_players, out_max_players = 0, 0
    # for file in tqdm(files, colour='green'): 
    #     in_players, out_players = get_max_positions(file)
    #     if in_players > in_max_players: 
    #         in_max_players = in_players
    #     if out_players > out_max_players: 
    #         out_max_players = out_players
            
    # print("Maximum number of input players in one play for all week, games, and plays: {}".format(in_max_players))
    # print("Maximum number of output players in one play for all week, games, and plays: {}".format(out_max_players))
    # exit()

    
    
    # for file in tqdm(files, colour='green'): 
    #     if 'input' in file: 
    #         file_dict = get_n_positions(file)
    #         dict1 = {k: max(N_POSITIONS[k], file_dict.get(k, N_POSITIONS[k])) for k in N_POSITIONS.keys()}
    #         N_POSITIONS = dict1
    #         dict1 = {}
    
    # print(N_POSITIONS)
    # exit()
    
    # pos = []
    # with mp.Pool(processes=mp.cpu_count()) as pool: 
    #     pos = pool.map(func=get_pos, iterable=files)
    
    # pos = list(set(chain.from_iterable(pos)))
    
    # transform(files=files)
        
    # with mp.Pool(processes=mp.cpu_count()) as pool: #TODO: Add functionality that enables to not open the file if it already exists
    #     func = partial(get_min_max, features=features)
    #     results = pool.map(func=func, iterable=files) 
    
    # dir_path, filename = os.path.split(min_max_path)
    # min_max_path = os.path.join(dir_path, f"updated_{filename}")
    
    # min_max_dict = {
    #     feature: 
    #         {
    #            'min': min((result[feature][0] for result in results), default=math.inf), 
    #            'max': max((result[feature][1] for result in results), default=-math.inf), 
    #         }
    #     for feature in results[0]
    # }
    
    # merged_min_max = dict(min_max_conf)
    # merged_min_max.update(min_max_dict)
    # oc.save(merged_min_max, min_max_path)
    
    positions = pos2num(file_path=args.pos_path)
    pos_path = os.path.join(os.getcwd(), args.pos_path)
    s = '\n'.join([f'{s[0]}: {str(s[1])}' for s in positions.items()])
    
    # TODO: Ensure that file can be overwritten correctly every time the encoding changes
    if os.path.exists(pos_path):
        with open(pos_path, 'w') as f: 
            f.truncate()
            f.write(s)
    else: 
        with open(pos_path, 'x') as f: 
            f.write(s)
    

if __name__ == '__main__': 
    main() 
