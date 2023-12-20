import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import SegmentDM
from src.lightning_module import SegmentModule


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):
    datamodule = SegmentDM(config.data_config)
    model = SegmentModule(config)

    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())

    experiment_save_path = Path(EXPERIMENTS_PATH) / config.experiment_name
    experiment_save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        str(experiment_save_path),
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=config.log_every_epochs,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=10, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(100, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
