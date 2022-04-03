import argparse
import yaml
from dataloader import split_ds, train_transforms, val_transforms, spleen_ds
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os.path as osp
from tools import TrainingModule
from monai.data import Dataset, ThreadDataLoader, list_data_collate, DataLoader, pad_list_data_collate
import torch
import warnings

warnings.filterwarnings('ignore')


def main(epochs, batch_size, output_dir, num_workers, buffer_size):
    # log
    log_dir = osp.join(output_dir, 'logs')
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=config['name'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_dice",
        filename="{epoch:03d}-{val_mean_dice:.4f}",
        save_last=True,
        save_top_k=3,
        mode="max",
        save_on_train_epoch_end=True
    )
    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        patience=20,
        mode='min'
    )

    # data
    train_dict, val_dict, test_dict = split_ds(data_config['dataset_dir'], 0.7, 0.2)
    # train_dict, val_dict, test_dict = spleen_ds(data_config['dataset_dir'], 0.8)
    train_transforms.set_random_state(seed)
    val_transforms.set_random_state(seed)
    train_ds = Dataset(train_dict, train_transforms)
    val_ds = Dataset(val_dict, val_transforms)
    test_ds = Dataset(test_dict, val_transforms)
    train_loader = ThreadDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                    buffer_size=buffer_size)
    val_loader = ThreadDataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2,
                                  collate_fn=pad_list_data_collate,
                                  buffer_size=4)

    test_loader = ThreadDataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers,
                                   collate_fn=pad_list_data_collate,
                                   buffer_size=2)

    # model
    model = TrainingModule(net_config)

    # train & val
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=epochs,
        logger=tb_logger,
        num_sanity_val_steps=0,
        checkpoint_callback=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=[lr_monitor, ModelSummary(max_depth=-1), checkpoint_callback, early_stop_callback]
    )

    if args.ckpt is None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.test(model, test_loader, ckpt_path=args.ckpt)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser('MISNet')
    parser.add_argument('--config', default='config.yaml')
    # parser.add_argument('--ckpt', default='logs/logs/R-UNet-0329/version_45/checkpoints/last.ckpt')
    parser.add_argument('--ckpt', default=None)
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    train_config = config['train_config']
    net_config = config['net_config']
    seed = config['seed']
    net_config['seed'] = seed
    main(**train_config)
