import os
import polars as pl
from tqdm import tqdm 

def main(): 
    data_dir = "/home/ehre/Documents/Projects/footballProject/data/nfl-big-data-bowl-2026-prediction/train"
    txt = os.path.join(os.getcwd(), "player_names.txt")
        
    for file in tqdm(os.listdir(data_dir)): 
        if "output" in file: 
            continue
        
        df = pl.read_csv(os.path.join(data_dir, file))
        players = df["player_name"].unique().to_list() 
          
        with open(txt, "w") as hf: 
            for player in players:    
                hf.write(f"{player}\n")


if __name__ == "__main__": 
    main()