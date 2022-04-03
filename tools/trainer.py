import torch
import pytorch_lightning as pl
from model import R_UNet, ResUnet
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai import transforms
from monai.optimizers import Novograd
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import numpy as np
from medpy import metric


def get_eval(y_hat, y):
    """
    dice：计算两幅图像中物体之间的Dice系数
    precision：  预测正确的个数占总的正类预测个数的比例（从预测结果角度看，有多少预测是准确的）
    recall：     确定了正类被预测为正类占所有标注的个数（从标注角度看，有多少被召回）
    tnr：        真负类率(True Negative Rate)：所有真实负类中，模型预测正确分类的比例
    """
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    dice = metric.dc(y_hat, y)
    precision = metric.precision(y_hat, y)
    recall = metric.recall(y_hat, y)
    tnr = metric.true_negative_rate(y_hat, y)

    return dice, precision, recall, tnr


class TrainingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.roi_size = self.config['slice_window_roi']
        self.model = R_UNet(config['seed'])
        self.loss_fn = DiceCELoss(to_onehot_y=True, sigmoid=True)
        self.post_pred = transforms.Compose([
            transforms.EnsureType(), transforms.Activations(sigmoid=True),
            transforms.AsDiscrete(threshold_values=True, threshold=0.5)
        ])
        self.post_label = transforms.Compose([
            transforms.EnsureType(), transforms.AsDiscrete(to_onehot=self.config['classes'])])  # 后处理标签
        # 在tensorboard画图不需要独热编码
        self.draw = transforms.Compose([
            transforms.EnsureType(), transforms.Activations(softmax=True), transforms.AsDiscrete(argmax=True)])

        # 在多分类的情况下 需要两个指标 一个算类的平均dice 一个算每一个类的dice
        self.train_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)  # 不算背景求dice
        self.val_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        # self.dice_metric_class = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    def training_step(self, batch, batch_idx):
        image, label = batch['image'], batch['label']
        result = self.model(image)
        loss = []
        dice = []
        for index in range(result[0]):
            loss, dice, precision, recall, tnr = self.shared_step(y_hat=result[index], y=label[index])
            self.log('train_loss', loss)
            self.log('train_dice', dice, prog_bar=True)
            self.logger.experiment.add_scalars('train_eval', {'precision': precision,
                                                              'recall': recall,
                                                              'tnr': tnr},
                                               global_step=self.current_epoch)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=False)
        return {'loss': loss, 'dice': dice}

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        sw_batch = 4
        result = sliding_window_inference(
            images, self.roi_size, sw_batch, self.model, mode='gaussian', overlap=0.2
        )
        loss, dice, precision, recall, tnr = self.shared_step(y_hat=result, y=labels)
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        outputs = [self.draw(i) for i in decollate_batch(result)]

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
        self.logger.experiment.add_scalars('val_eval', {'precision': precision,
                                                        'recall': recall,
                                                        'tnr': tnr},
                                           global_step=self.current_epoch)

        plot_2d_or_3d_image(tag="result", data=outputs, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=images, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        return {'val_loss': loss, 'val_dice': dice}

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        sw_batch = 4
        result = sliding_window_inference(
            images, self.roi_size, sw_batch, self.model, mode='gaussian', overlap=0.2)
        loss, dice, precision, recall, tnr = self.shared_step(y_hat=result, y=labels)
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        outputs = [self.draw(i) for i in decollate_batch(result)]

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_dice', dice, prog_bar=True)
        self.logger.experiment.add_scalars('test_eval', {'precision': precision,
                                                         'recall': recall,
                                                         'tnr': tnr},
                                           global_step=self.current_epoch)

        plot_2d_or_3d_image(tag="result", data=outputs, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=images, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        return {'test_loss': loss, 'test_dice': dice}

    def training_epoch_end(self, step_outputs):
        losses, dice = self.shared_epoch_end(step_outputs, 'loss', 'dice')
        self.log('train_mean_loss', losses)
        self.log('train_mean_dice', dice)

    def validation_epoch_end(self, step_outputs):
        losses, dice = self.shared_epoch_end(step_outputs, 'val_loss', 'val_dice')
        self.log('val_mean_loss', losses)
        self.log('val_mean_dice', dice)

    def test_epoch_end(self, step_outputs):
        losses, dice = self.shared_epoch_end(step_outputs, 'test_loss', 'test_dice')
        self.log('test_mean_loss', losses)
        self.log('test_mean_dice', dice)

    def shared_epoch_end(self, outputs, loss_key, dice_key):
        losses = []
        dices = []
        for output in outputs:
            loss = output[loss_key].item()
            losses.append(loss)
            dice = output[dice_key]
            dices.append(dice)

        losses = float(np.mean(losses))
        dices = float(np.mean(dices))

        return losses, dices

    def shared_step(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)
        loss = torch.nan_to_num(loss)
        dice, precision, recall, tnr = get_eval(y_hat.cpu().detach().numpy(), y.cpu().detach().numpy())

        return loss, dice, precision, recall, tnr

    def configure_optimizers(self):
        # scheduler 在鞍点的时候减少学习率
        # optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        optimizer = Novograd(self.model.parameters(), self.learning_rate * 10)
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
