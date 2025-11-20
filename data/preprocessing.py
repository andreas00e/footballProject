import os 
import math 
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path
import multiprocessing as mp 
from functools import partial
from typing import List, Literal
from omegaconf import OmegaConf as oc


def pos_to_num(file_path: Path) -> dict[str: float]: 
    """ 
    Transform position labels into numeric form for network processing
    
    Returns:
        dict[str: float]: Dictionary the keys being the players and the values being the numeric position encodings
    """
    with open(file_path) as f: 
        pos_lit = [pos.replace(' ', '') for pos in f.read().split('\n')]
        pos_num = list(map(lambda x: math.sin(math.sqrt(int.from_bytes(x.encode('utf-8'), 'big'))), pos_lit))
        
        return dict(zip(pos_lit, pos_num))

def get_min_max(file: Path, features: List[str]): # min_max_conf is a DictConfig...
    data = pd.read_csv(file)
    data['player_height'] = data['player_height'].apply(lambda x: (int(x.split('-')[0])*12.0+int(x.split('-')[1]))*2.54)
    min_max_dict = dict(zip(features, zip(data[features].min(), data[features].max())))
    
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
    data_path = train_conf['data']['dataset']['data_path']
    # files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv') and 'input' in file]
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv')]
    features = train_conf['features_of_interest']['model']['norm']
    
    transform(files=files)
    exit()
        
    #TODO: Add functionality that enables to not open the file if it already exists
    with mp.Pool(processes=mp.cpu_count()) as pool: 
        func = partial(get_min_max, features=features)
        results = pool.map(func=func, iterable=files) 
    
    # Old implementation from 04.11.25  
    # min_max_path = '/'.join(min_max_path.split('/')[:-1])+'/updated_'+min_max_path.split('/')[-1]
    dir_path, filename = os.path.split(min_max_path)
    min_max_path = os.path.join(dir_path, f"updated_{filename}")
    
    min_max_dict = {
        feature: 
            {
               'min': min((result[feature][0] for result in results), default=math.inf), 
               'max': max((result[feature][1] for result in results), default=-math.inf), 
            }
        for feature in results[0]
    }
    
    merged_min_max = dict(min_max_conf)
    merged_min_max.update(min_max_dict)
    oc.save(merged_min_max, min_max_path)
    
    positions = pos_to_num(file_path=args.pos_path)
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
