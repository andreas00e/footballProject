import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("add_2", lambda x, y: x + y)
OmegaConf.register_new_resolver("add_3", lambda x, y, z: x + y + z)

import torch
from torch.utils.data import DataLoader, random_split
torch.set_float32_matmul_precision("medium")

import lightning as L 
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False # suppress SLURM warning

from models.models import TransformerModel
from data.polars_data_loading import PlayDataset
from data.utils import collate_fn

@hydra.main(config_path="./conf", config_name="train", version_base=None)
def main(cfg: DictConfig): 
    L.seed_everything(**cfg.seed_everything)
    
    dataset = PlayDataset(**cfg.data.dataset, feature_config=cfg.feature_config, data_type=cfg.data.data_type)
         
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=cfg.data.dataloading.lengths)
    del cfg.data.dataloading.lengths
    train_dataloader = DataLoader(dataset=train_dataset, **cfg.data.dataloading, collate_fn=collate_fn)
    
    del cfg.data.dataloading.shuffle
    val_dataloader = DataLoader(dataset=val_dataset, **cfg.data.dataloading, collate_fn=collate_fn)  

    model = TransformerModel(feature_config=cfg.feature_config, size_window=cfg.model.window_size, 
                             transformer=cfg.model.transformer, in_emb=cfg.model.i, out_emb=cfg.model.o)
    logger = WandbLogger(**cfg.logger)
    modelCheckpoint = ModelCheckpoint(**cfg.modelCheckpoint)
    profiler = SimpleProfiler(**cfg.profiler)
    trainer  = L.Trainer(logger=logger, callbacks=[modelCheckpoint], profiler=profiler, **cfg.trainer)
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    
if __name__ == "__main__": 
    main() 