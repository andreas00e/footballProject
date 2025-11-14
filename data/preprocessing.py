import os 
import math 
import hydra
import argparse
import numpy as np 
import pandas as pd
from typing import List, Literal, Type
from pathlib import Path
import multiprocessing as mp 
from functools import partial
from omegaconf import OmegaConf as oc


FLIP_FEATURES = ['absolute_yardline_number', 'x', 'y', 'o', 'dir']

def pos_to_num(file_path: Path):
    with open(file_path) as f: 
        pos_lit = [pos.replace(' ', '') for pos in f.read().split('\n')]
        pos_num = list(map(lambda x: math.sin(math.sqrt(int.from_bytes(x.encode('utf-8'), 'big'))), pos_lit))
        return dict(zip(pos_lit, pos_num))

def get_min_max(file: Path, features: List[str]): # min_max_conf is a DictConfig...
    data = pd.read_csv(file)
    data['player_height'] = data['player_height'].apply(lambda x: (int(x.split('-')[0])*12.0+int(x.split('-')[1]))*2.54)
    min_max_dict = dict(zip(features, zip(data[features].min(), data[features].max())))
    return min_max_dict

type axis = Literal['⇅', '⇄', 'both']
def transform(files: List[Path], flip: axis) -> None: 
    for file in files: 
        data = pd.read_csv(file)
        file_path, file_name = os.path.split(file)
        file_path_inv = file_path+'_inv'
        file_name_inv = '_'.join(file_name.split('.')[0], 'inv.csv')
        
        data_inv = {}   
        match flip: 
            case '⇅': # flip along x-axis 
                data['x_⇅'] = 120.0 - data['x'] 
                data[['o_⇅', 'dir_⇅']] = 180.0 - data[['o', 'dir']] 
            case '⇄': # flip along y-axis
                data['y_⇄'] = 53.3 - data['y'] 
                data[['o_⇄', 'dir_⇄']] = 360.0 - data[['o', 'dir']] 
            case 'both': # flip along both x-axis and y-axis
                data_inv['x_inv'] = list(120.0 - data['x'])
                data_inv['y_inv'] = list(53.3 - data['y'])
                data_inv = {}   
                if 'input' in file: 
                    data_inv['o_inv'] = list((data['o']+180.0)%360.0)
                    data_inv['dir_inv'] = list((data['dir']+180.0)%360.0)
                    data_inv['player_direction_inv'] = list(int(not data['player_direction']))
                    data_inv['playe_side_inv'] = list(int(not data['player_side_direction']))
                    
        data_inv = pd.DataFrame(data=data_inv)
        data_inv = data_inv.to_csv(file_name='file.csv', file_path=data_inv_path) # TODO: Make sure file contains 'input'/'output' and week 
                
        

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
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv') and 'input' in file]
    features = train_conf['features_of_interest']['model']['norm']
    
    transform(files=files, flip='horizontal')
    
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
    
    # Old implementation from 04.11.25
    # for feature in features: 
    #     for result in results: 
    #         if isinstance(min_max_conf[feature]['min'], type(None)): 
    #             min_max_conf[feature]['min'] = math.inf
    #         if isinstance(min_max_conf[feature]['max'], type(None)): 
    #             min_max_conf[feature]['max'] = -math.inf
    #         if result[feature][0] < min_max_conf[feature]['min']: 
    #             min_max_conf[feature]['min'] = result[feature][0]
    #         if result[feature][1] > min_max_conf[feature]['max']: 
    #             min_max_conf[feature]['max'] = result[feature][1]
    #     oc.save(min_max_conf, min_max_path)
            

if __name__ == '__main__': 
    main() 
