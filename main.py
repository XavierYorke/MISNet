import argparse
import yaml
from dataloader import split_ds, train_transforms, val_transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os.path as osp
from tools import TrainingModule
from monai.data import Dataset, ThreadDataLoader, list_data_collate, DataLoader, CacheDataset
import torch


def main(epochs, batch_size, output_dir, num_workers, buffer_size):
    # log
    log_dir = osp.join(output_dir, 'logs')
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=config['name'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_dice",
        filename="{epoch:02d}-{dice:.4f}",
        save_last=True,
        save_top_k=3,
        mode="max",
        save_on_train_epoch_end=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # data
    train_dict, val_dict = split_ds(data_config['dataset_dir'], 0.8)
    train_transforms.set_random_state(seed)
    val_transforms.set_random_state(seed)
    train_ds = Dataset(train_dict, train_transforms)
    val_ds = Dataset(val_dict, val_transforms)
    train_loader = ThreadDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                    buffer_size=buffer_size)
    val_loader = ThreadDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  buffer_size=buffer_size)

    # model
    model = TrainingModule(net_config)

    # train & val
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=epochs,
        logger=tb_logger,
        num_sanity_val_steps=1,
        checkpoint_callback=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=[lr_monitor, ModelSummary(max_depth=-1), checkpoint_callback, early_stop_callback]
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser('MISNet')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    train_config = config['train_config']
    net_config = config['net_config']
    seed = config['seed']
    net_config['seed'] = seed
    main(**train_config)
