import csv
import monai.optimizers
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    Activations
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from pytorch_lightning.utilities.seed import seed_everything
from monai.data import decollate_batch, write_nifti
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.data.image_reader import NibabelReader
from monai.data import list_data_collate, ThreadDataLoader, Dataset
from dataloader import train_trans, val_trans
import pytorch_lightning
import os
import torch.nn.init
from typing import Any, Callable

import torch
import torch.nn as nn
from monai.networks.layers import Norm
import numpy as np


def kaiming_normal_init(
        m, normal_func: Callable[[torch.Tensor, float, float], Any] = torch.nn.init.kaiming_normal_) -> None:
    cname = m.__class__.__name__

    if getattr(m, "weight", None) is not None and (cname.find("Conv") != -1 or cname.find("Linear") != -1):
        normal_func(m.weight.data)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif cname.find("BatchNorm") != -1:
        normal_func(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class MISNet(pytorch_lightning.LightningModule):
    def __init__(self, log_dir, classes, batch_size, learning_rate, slice_window_roi, predict_root, data_dir):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            # channels=(32, 64, 128, 256, 512),
            channels=(1, 1, 1, 1, 1),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE,
        )
        self.classes = classes
        self.log_dir = log_dir
        self.predict_root = predict_root
        self.data_dir = data_dir

        # 复现种子点
        set_determinism(seed=0)
        seed_everything(seed=0)

        # 网络初始化
        # self._model.apply(kaiming_normal_init)
        self._model.apply(kaiming_normal_init)

        self.loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True)  # 简单损失函数 softmax激活
        # self.loss_function = DiceFocalLoss(to_onehot_y=True,sigmoid=True,focal_weight=[2,4,4])
        # 多分类任务后处理逻辑 先softmax 激活取 argmax 然后one 按通道编码计算dice指标
        self.post_pred = Compose([EnsureType(), Activations(softmax=True),
                                  AsDiscrete(argmax=True, to_onehot=self.classes)])  # 先argmax 再to one hot 最后threhold
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=self.classes)])  # 后处理标签
        # 在tensorboard画图不需要独热编码
        self.draw = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True)])

        # 在多分类的情况下 需要两个指标 一个算类的平均dice 一个算每一个类的dice
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)  # 不算背景求dice 指标
        self.dice_metric_class = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)  #

        self.best_val_dice = 0
        self.best_val_epoch = 0

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.roi_size = slice_window_roi
        self.reader = NibabelReader()

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path
        files_path = os.listdir(self.data_dir)
        image_paths = []
        label_paths = []
        for file_path in files_path:
            file = os.path.join(self.data_dir, file_path)
            image_path = os.path.join(file, file_path + '_origin.nii.gz')
            label_path = os.path.join(file, file_path + '_ias.nii.gz')
            image_paths.append(image_path)
            label_paths.append(label_path)

        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(image_paths, label_paths)
        ]

        # 训练集 测试集 验证机集 划分
        train_files, val_files, test_files = data_dicts[:-20], data_dicts[-20:], data_dicts[-20:]
        # define the data transforms
        train_transforms = train_trans
        val_transforms = val_trans

        train_transforms.set_random_state(seed=0)
        val_transforms.set_random_state(seed=0)

        self.train_ds = Dataset(data=train_files, transform=train_transforms)

        self.val_ds = Dataset(data=val_files, transform=val_transforms)

        self.test_ds = Dataset(data=test_files, transform=val_transforms)

    # 定义相关loader 为了异步加载使用 ThreadDataLoader
    def train_dataloader(self):
        train_loader = ThreadDataLoader(self.train_ds, buffer_size=12, num_workers=6,
                                        batch_size=self.batch_size,
                                        shuffle=True, collate_fn=list_data_collate)
        return train_loader

    def val_dataloader(self):
        val_loader = ThreadDataLoader(
            self.val_ds, batch_size=1, num_workers=0)
        return val_loader

    def test_dataloader(self):
        test_loader = ThreadDataLoader(
            self.test_ds, batch_size=1, num_workers=0)
        return test_loader

    def configure_optimizers(self):
        # scheduler 在鞍点的时候减少学习率
        # optimizer = torch.optim.Adam(self._model.parameters(), self.learning_rate)
        optimizer = monai.optimizers.Novograd(self._model.parameters(), self.learning_rate * 10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10,
                                                               min_lr=1e-6,
                                                               verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': "train_loss"
        }

    # 每个 traning_step
    def training_step(self, batch, batch_idx):
        # training_step_start = time()
        images, labels = batch["image"], batch["label"]

        output = self.forward(images)
        loss = self.loss_function(output, labels)
        # 调用self.log记录指标
        self.log("train_loss", loss.item(), logger=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=False)

        # training_step_end = time()
        # print("traning_step_time",training_step_end-training_step_start)
        return {"loss": loss}

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.roi_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward, mode="gaussian")

        loss = self.loss_function(outputs, labels)
        outputs_for_dice = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs_for_dice, y=labels)
        self.dice_metric_class(y_pred=outputs_for_dice, y=labels)
        outputs = [self.draw(i) for i in decollate_batch(outputs)]
        # 画图对比验证
        plot_2d_or_3d_image(tag="result", data=outputs, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=images, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        return {"val_loss": loss, "val_number": len(outputs)}

    # 验证结束 更新指标
    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        dice_mertric = self.dice_metric
        dice_mertric_class = self.dice_metric_class
        mean_val_dice = dice_mertric.aggregate().item()
        dice_mertric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        class_val_dice = dice_mertric_class.aggregate()
        channel_dice = [i.item() for i in class_val_dice]
        dice_mertric.reset()
        print("classes dice", channel_dice)
        self.log("mean_val_dice", mean_val_dice, prog_bar=True)
        self.log("mean_val_loss", mean_val_loss, prog_bar=True)
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )

        return

    def write_csv(self, csv_name, content, mul=True, mod="w"):
        """write list to .csv file."""
        with open(csv_name, mod) as myfile:
            writer = csv.writer(myfile)
            if mul:
                writer.writerows(content)
            else:
                writer.writerow(content)

    # 自定义optimizer_zero_grad pytorch官方说的加速方法
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # zero_grad_start=time()
        optimizer.zero_grad(set_to_none=True)
        # zero_grad_end = time()
        # print("zero grad time",zero_grad_end-zero_grad_start)

    def on_test_epoch_start(self) -> None:
        # CSV HEAD
        self.out_total_csv_path = os.path.join(self.log_dir, 'total_seg_result.csv')

        out_content = ["name"]

        for i in range(0, self.classes):
            out_content.extend(["class_" + str(i)])

        self.write_csv(self.out_total_csv_path, out_content, mul=False, mod='a+')

    # 测试test_step
    def test_step(self, batch, batch_idx):

        images, labels = batch["image"], batch["label"]

        file_name = os.path.basename(batch['image_meta_dict']['filename_or_obj'][0])
        roi_size = self.roi_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward, mode="gaussian"
        )
        outputs_dice = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        draw = [self.draw(i) for i in decollate_batch(outputs)]
        self.dice_metric(y_pred=outputs_dice, y=labels)
        self.dice_metric_class(y_pred=outputs_dice, y=labels)

        class_val_dice = self.dice_metric_class.aggregate()
        channel_dice = [i.item() for i in class_val_dice]
        # 保存结果
        print(file_name + "_class_dice:", channel_dice)

        out_content = [os.path.basename(batch['image_meta_dict']['filename_or_obj'][0])]
        for i in range(0, len(channel_dice)):
            out_content.extend([channel_dice[i]])

        data_path = batch['image_meta_dict']['filename_or_obj'][0]

        data = self.reader.read(data_path)
        _, _meta = self.reader.get_data(data)

        write_nifti(data=draw[0].squeeze(0), file_name=os.path.join(self.predict_root, file_name),
                    output_dtype=np.int16, resample=False,
                    output_spatial_shape=_meta['spatial_shape'],
                    affine=_meta['affine'])
        self.write_csv(self.out_total_csv_path, out_content, mul=False, mod='a+')

    # 测试结束
    def test_epoch_end(self, outputs):
        dice_mertric = self.dice_metric
        dice_mertric_class = self.dice_metric_class
        mean_test_dice = dice_mertric.aggregate().item()
        chanel_mertric_class = dice_mertric_class.aggregate()
        chanel_mertric_class = [i.item() for i in chanel_mertric_class]
        print("test_set_mean dice", mean_test_dice)
        print("test_set_class dice", chanel_mertric_class)
        dice_mertric.reset()
        dice_mertric_class.reset()
