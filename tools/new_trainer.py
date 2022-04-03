import torch
import pytorch_lightning as pl
from model import R_UNet, ResUnet
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, ConfusionMatrixMetric
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
        self.metrics = DiceMetric(
            include_background=False, reduction="mean_batch")
        self.hd_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean", get_not_nans=False)
        self.sd_metric = SurfaceDistanceMetric(
            include_background=False, reduction="mean", get_not_nans=False)
        # precision recall 这个混淆矩阵可以写在一起，分开写精确一点，知道谁是谁
        self.precision_metric = ConfusionMatrixMetric(metric_name="precision", include_background=False,
                                                      reduction="mean", get_not_nans=False, compute_sample=True)
        self.recall_metric = ConfusionMatrixMetric(metric_name="recall", include_background=False, reduction="mean",
                                                   get_not_nans=False, compute_sample=True)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = self.model(x)

        loss, dice, hd, sd, precision, recall = self.shared_step(y_hat=y_hat, y=y)
        self.log('train_loss', loss)
        self.log('train_dice', dice, prog_bar=True)
        self.log('train_hd', hd)
        self.log('train_sd', sd)
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.logger.experiment.add_scalars('train_eval', {'precision': precision,
                                                          'recall': recall,
                                                          'hd': hd,
                                                          'sd': sd},
                                           global_step=self.current_epoch)
        return {'loss': loss, 'dice': dice}

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = sliding_window_inference(x, roi_size=self.roi_size, sw_batch_size=4,
                                         predictor=self.model, mode='gaussian',
                                         overlap=self.config['overlap'])
        loss, dice, hd, sd, precision, recall = self.shared_step(y_hat=y_hat, y=y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
        self.log('val_hd', hd)
        self.log('val_sd', sd)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.logger.experiment.add_scalars('val_eval', {'precision': precision,
                                                        'recall': recall,
                                                        'hd': hd,
                                                        'sd': sd},
                                           global_step=self.current_epoch)
        labels = [self.post_label(i) for i in decollate_batch(y)]
        outputs = [self.draw(i) for i in decollate_batch(y_hat)]
        plot_2d_or_3d_image(tag="result", data=outputs, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=x, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        return {'loss': loss, 'dice': dice}

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = sliding_window_inference(x, roi_size=self.roi_size, sw_batch_size=4, predictor=self.model,
                                         overlap=self.config['overlap'], mode='gaussian')

        loss, dice, hd, sd, precision, recall = self.shared_step(y_hat=y_hat, y=y)
        self.logger.experiment.add_scalars('test_eval', {'precision': precision,
                                                         'recall': recall,
                                                         'hd': hd,
                                                         'sd': sd},
                                           global_step=self.current_epoch)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_dice', dice, prog_bar=True)
        self.log('test_hd', hd)
        self.log('test_sd', sd)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        return {'loss': loss, 'dice': dice}

    def training_epoch_end(self, outputs):
        losses, dice = self.shared_epoch_end(outputs, 'loss')
        self.log('train_mean_loss', float(losses))
        self.log('train_mean_dice', float(dice))

    def validation_epoch_end(self, outputs):
        losses, dice = self.shared_epoch_end(outputs, 'loss')
        self.log('val_mean_loss', float(losses))
        self.log('val_mean_dice', float(dice))

    def test_epoch_end(self, outputs):
        losses, dice = self.shared_epoch_end(outputs, 'loss')
        self.log('test_mean_loss', float(losses), prog_bar=True)
        self.log('test_mean_dice', float(dice), prog_bar=True)

    def shared_epoch_end(self, outputs, loss_key):
        losses = []
        dices = []
        for output in outputs:
            loss = output[loss_key].item()
            losses.append(loss)
            dice = output['dice']
            dices.append(dice)

        losses = np.mean(np.array(losses))
        dice = np.mean(np.array(dices))


        # dice = dice.detach().cpu().numpy()
        # hd = hd.detach().cpu().numpy()
        # sd = sd.detach().cpu().numpy()
        # precision = precision.detach().cpu().numpy()
        # recall = recall.detach().cpu().numpy()
        return losses, dice     # , hd, sd, precision, recall

    def shared_step(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)

        y_hat = [self.post_pred(it) for it in decollate_batch(y_hat)]
        y = decollate_batch(y)

        dice = torch.mean(self.metrics(y_hat, y)).item()
        self.hd_metric(y_hat, y)
        self.sd_metric(y_hat, y)
        self.precision_metric(y_hat, y)
        self.recall_metric(y_hat, y)

        hd = self.hd_metric.aggregate().item()
        sd = self.sd_metric.aggregate().item()
        precision = self.precision_metric.aggregate()[0].item()
        recall = self.recall_metric.aggregate()[0].item()

        self.metrics.reset()
        self.hd_metric.reset()
        self.sd_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()

        return loss, dice, hd, sd, precision, recall

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
