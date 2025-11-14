import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader, random_split

import hydra
import lightning as L 
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler

from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False # suppress SLURM warning from: https://github.com/Lightning-AI/pytorch-lightning/issues/6389

from models.models import TransformerModel
from data.data_loading import SequentialDataset

@hydra.main(config_path='./conf', config_name='train', version_base=None)
def main(cfg): 
    L.seed_everything(**cfg.seed_everything)
    
    dataset = SequentialDataset(**cfg.data.dataset, f_o_i=cfg.features_of_interest)
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=cfg.data.dataloading.lengths)

    del cfg.data.dataloading.lengths
    train_dataloader = DataLoader(dataset=train_dataset, **cfg.data.dataloading)
    del cfg.data.dataloading.shuffle
    val_dataloader = DataLoader(dataset=val_dataset, **cfg.data.dataloading)
        
    model = TransformerModel(f_o_i=cfg.features_of_interest, transformer=cfg.model.transformer, in_emb=cfg.model.i, out_emb=cfg.model.o)
    logger = WandbLogger(**cfg.logger)
    modelCheckpoint = ModelCheckpoint(**cfg.modelCheckpoint)
    profiler = SimpleProfiler(**cfg.profiler)
    trainer  = L.Trainer(logger=logger, callbacks=[modelCheckpoint], profiler=profiler, **cfg.trainer)
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    

if __name__ == '__main__': 
    main() 