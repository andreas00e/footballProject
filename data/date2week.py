import os 
import csv
import hydra
import polars as pl 

@hydra.main(config_path="./confs", config_name="run", version_base=None)
def main(cfg): 
    dir_path = cfg.data.dataset.data_dir.replace("_test", "")
    files = [file for file in os.listdir(dir_path) if "input" in file]
    df = {}
    
    for file in files: 
        week = file.split("_")[-1].split(".")[0].replace("w", "")
        frame = pl.read_csv(os.path.join(dir_path, file))
        dates = frame["game_id"].unique().to_list() 
        dates = sorted([int(str(date)[4:8]) for date in dates])
        start, end = dates[0], dates[-1]
        
        df[week] = [start, end]
    
    df = pl.from_dicts(df, schema=[a[-2:] for a in df.keys()])
    df = df.with_row_index("weeks").sort(by="weeks")
    names = df.rows
    print(names)
    # df = df.with_columns(pl.Series("weeks", ["start", "end"]).select(["weeks", *df.columns]))
    df = df.write_csv("./date2week")
        
    # print(df)    

if __name__ == "__main__": 
    main() 
    