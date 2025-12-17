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

from models.models import TransformerModel
from data.polars_data_loading import PlayDataset
from data.utils import collate_fn

@hydra.main(config_path="./conf", config_name="train", version_base=None)
def main(cfg: DictConfig): 
    L.seed_everything(**cfg.seed_everything)
    
    dataset = PlayDataset(**cfg.data.dataset, feature_config=cfg.feature_config, data_type=cfg.data.data_type)
         
    del cfg.data.dataloading.lengths
    dataloader = DataLoader(dataset=dataset, **cfg.data.dataloading, collate_fn=collate_fn)

        
    logger = WandbLogger(**cfg.logger)
    model = TransformerModel.load_from_checkpoint(checkpoint_path="logs/checkpoint_model/WhatAModelTheBestModelEverybodySaysThat.ckpt", weights_only=False)
    trainer  = L.Trainer(logger=logger, **cfg.trainer)
    
    trainer.test(model=model, dataloaders=dataloader, weights_only=False)
    
    
if __name__ == "__main__": 
    main() 