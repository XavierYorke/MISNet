import torch
import pytorch_lightning as pl
from model import R_UNet
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai import transforms
from monai.optimizers import Novograd
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import numpy as np


class TrainingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.roi_size = self.config['slice_window_roi']
        # self.reader = NibabelReader()
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
        loss, dice = self.shared_step(y_hat=result, y=label, mode='train')
        # loss = self.loss_fn(result, label)
        # log打印的是均值，return的是每一次的
        self.log('train_loss', loss)
        self.log('train_dice', dice, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        sw_batch = 4
        result = sliding_window_inference(
            images, self.roi_size, sw_batch, self.model, mode='gaussian'
        )
        # loss = self.loss_fn(result, labels)
        loss, dice = self.shared_step(y_hat=result, y=labels, mode='val')
        # outputs_for_dice = [self.post_pred(i) for i in decollate_batch(result)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        # self.dice_metric(y_pred=outputs_for_dice, y=labels)
        # self.dice_metric_class(y_pred=outputs_for_dice, y=labels)
        outputs = [self.draw(i) for i in decollate_batch(result)]

        self.log('val_loss', loss)
        self.log('val_dice', dice)

        plot_2d_or_3d_image(tag="result", data=outputs, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=images, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        # return {'val_loss': loss, 'val_number': len(outputs)}
        # return {'val_loss': loss, 'val_dice': dice}
        return {'val_loss': loss}

    def training_epoch_end(self, step_outputs):
        # loss = 0.
        # for result in step_outputs:
        #     loss += result['loss']
        # loss = loss / len(step_outputs)

        losses, dice = self.shared_epoch_end(step_outputs, 'loss')
        self.log('train_mean_loss', losses)
        self.log('train_mean_dice', dice)

    def validation_epoch_end(self, step_outputs):
        # val_loss, num_items = 0, 0
        # for output in step_outputs:
        #     val_loss += output["val_loss"].sum().item()
        #     num_items += output["val_number"]
        #
        # dice_mertric = self.dice_metric
        # dice_mertric_class = self.dice_metric_class
        # mean_val_dice = dice_mertric.aggregate().item()
        # dice_mertric.reset()
        # mean_val_loss = torch.tensor(val_loss / num_items)
        # class_val_dice = dice_mertric_class.aggregate()
        # channel_dice = [i.item() for i in class_val_dice]
        # dice_mertric.reset()

        losses, dice = self.shared_epoch_end(step_outputs, 'val_loss')
        self.log('val_mean_loss', losses, prog_bar=True)
        self.log('val_mean_dice', dice, prog_bar=True)

    def shared_epoch_end(self, outputs, loss_key):
        losses = []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key].item()
            losses.append(loss)

        losses = np.array(losses)
        losses = np.mean(losses)

        if loss_key == 'loss':
            dice = self.train_metric.aggregate().item()
            self.train_metric.reset()
        if loss_key == 'val_loss':
            dice = self.val_metric.aggregate().item()
            self.val_metric.reset()

        return losses, dice

    def shared_step(self, y_hat, y, mode):
        loss = self.loss_fn(y_hat, y)

        y_hat = [self.post_pred(it) for it in decollate_batch(y_hat)]
        y = decollate_batch(y)

        if mode == 'train':
            dice = self.train_metric(y_hat, y)
        if mode == 'val':
            dice = self.val_metric(y_hat, y)
        dice = torch.nan_to_num(dice)
        loss = torch.nan_to_num(loss)

        dice = torch.mean(dice, dim=0)
        return loss, dice

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
