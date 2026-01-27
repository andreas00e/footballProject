import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("add_2", lambda x, y: x + y)
OmegaConf.register_new_resolver("add_3", lambda x, y, z: x + y + z)

import torch
torch.set_float32_matmul_precision("medium")
from torch.utils.data import DataLoader

import lightning as L 

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False # suppress SLURM warning

from models.models_copy import DecoderOnlyTransformer
from data.polars_data_loading import PlayDataset
from data.utils import collate_fn_graph

@hydra.main(config_path="./conf", config_name="inference", version_base=None)
def main(cfg: DictConfig):     
    dataset = PlayDataset(**cfg.data.dataset, feature_config=cfg.feature_config, data_type=cfg.data.data_type)
    
    dataloader = DataLoader(dataset=dataset, **cfg.data.dataloading, collate_fn=collate_fn_graph)
    
    model = DecoderOnlyTransformer.load_from_checkpoint(checkpoint_path="/home/ehre/Documents/Projects/footballProject/logs/checkpoint_graph_model/WhatAModelTheBestModelEverybodySaysThat.ckpt", weights_only=False)
    trainer = L.Trainer(**cfg.trainer)
    
    out = trainer.predict(model=model, dataloaders=dataloader)
    
    print("Shape of input: {}".format(next(iter(dataloader))["sources"].x.shape))
    print(f"Length of output: {len(out)}")
    print(f"Shape of output: {out[0].shape}")
    
if __name__ == "__main__": 
    main() 