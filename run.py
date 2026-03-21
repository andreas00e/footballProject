import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("add_2", lambda x, y: x + y)
OmegaConf.register_new_resolver("sub_2", lambda x, y: x - y)
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

from data.utils import collate_fn_graph
from data.polars_data_loading import PlayDataset
from models.models import DecoderOnlyTransformer


@hydra.main(config_path="./confs", config_name="run", version_base=None)
def main(cfg: DictConfig): 

    if cfg.mode == "train": 
        L.seed_everything(**cfg.seed_everything)
        dataset = PlayDataset(**cfg.data.dataset)
        train_dataset, val_dataset = random_split(dataset=dataset, lengths=cfg.data.random_split.lengths)
        train_dataloader = DataLoader(dataset=train_dataset, **cfg.data.dataloading, collate_fn=collate_fn_graph)
        del cfg.data.dataloading.shuffle
        val_dataloader = DataLoader(dataset=val_dataset, **cfg.data.dataloading, collate_fn=collate_fn_graph) 

        model = DecoderOnlyTransformer(**cfg.model)
        logger = WandbLogger(**cfg.logger)
        modelCheckpoint = ModelCheckpoint(**cfg.modelCheckpoint)
        profiler = SimpleProfiler(**cfg.profiler)
        trainer = L.Trainer(accelerator="cpu", devices=1, logger=logger, callbacks=[modelCheckpoint], profiler=profiler, **cfg.trainer)
        
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    if cfg.mode == "predict": 
        cfg.data.dataset.data_dir = cfg.prediction.test_dir
        dataset = PlayDataset(**cfg.data.dataset)
        dataloader = DataLoader(dataset=dataset, num_workers=cfg.data.dataloading.num_workers, collate_fn=collate_fn_graph)
        model = DecoderOnlyTransformer.load_from_checkpoint(checkpoint_path=cfg.prediction.checkpoint_path, weights_only=False)
        trainer = L.Trainer(**cfg.trainer)
        
        out = trainer.predict(model=model, dataloaders=dataloader)
        
        for idx, (y, y_hat) in enumerate(out): 
            print(f"Input at step {idx}: \n {y}")
            print(f"Shape of input ate step {idx}: {y.shape}")
            print("---------------------------------------------")
            print(f"Output at step {idx}: \n {y_hat}")
            print(f"Shape of output at step {idx}: {y_hat.shape}")

if __name__ == "__main__": 
    main() 