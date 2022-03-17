from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
import pytorch_lightning
import os
from model import MISNet


def train(resume, ckpt_path, log_dir, max_epochs, check_val_every_n_epoch,
          log_every_n_steps, precision, net_config):
    net = MISNet(log_dir, **net_config)
    # set up loggers and checkpoints
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 保存check point的规则
    # --monitor self.log 里边记录的指标
    # --filename 保存文件名
    # --save_last=True 以防万一保存最后一个epoch的checkpoint
    # --save_top_K 保存几个最好的模型
    # --mode 指定指标的大小，一般来说 loss是min acc那些是max
    # -- save_on_train_epoch_end epoch 结束后保存
    checkpoint_callback = ModelCheckpoint(
        monitor="mean_val_dice",
        filename="vessel_seg-{epoch:02d}-{mean_val_dice:.2f}",
        save_last=True,
        save_top_k=3,
        mode="max",
        save_on_train_epoch_end=True
    )

    # Trainer的设置
    # --gpus:指定的gpu:有多卡的话 可以写成[0,1]的形式
    # --max_epochs: 训练轮数
    # --logger ：logger的类型 monai用的是tensorboard类型
    # --checkpoint_callback:是否用call_back 处理checkpoint
    # --check_val_every_n_epoch：每隔几个epoch 验证一次
    # --log_every_n_steps ：记录指标setp的间隔
    # --precision ：混合精度
    # --callbacks : 定义的call_back 函数
    # --profiler="simple" :调优 profiler
    trainer = pytorch_lightning.Trainer(
        gpus=[0],
        max_epochs=max_epochs,
        logger=tb_logger,
        checkpoint_callback=True,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
        precision=precision,
        callbacks=[lr_monitor, ModelSummary(max_depth=-1), checkpoint_callback],
        # profiler="simple"
    )

    if resume:
        trainer.fit(net, ckpt_path=ckpt_path)
    else:
        trainer.fit(net)
    trainer.test(net)
