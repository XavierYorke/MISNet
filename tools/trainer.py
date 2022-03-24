import torch
import pytorch_lightning as pl
from model import R_UNet
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data.image_reader import NibabelReader
from monai import transforms
from monai.optimizers import Novograd
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch


class TrainingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.roi_size = self.config['slice_window_roi']
        self.reader = NibabelReader()
        self.model = R_UNet(config['seed'])
        self.loss_fn = DiceCELoss(to_onehot_y=True, sigmoid=True)
        # self.loss_function = DiceFocalLoss(to_onehot_y=True,sigmoid=True,focal_weight=[2,4,4])
        # 多分类任务后处理逻辑 先softmax 激活取 argmax 然后one 按通道编码计算dice指标
        self.post_pred = transforms.Compose([transforms.EnsureType(), transforms.Activations(softmax=True),
                                             transforms.AsDiscrete(argmax=True, to_onehot=self.config[
                                                 'classes'])])  # 先argmax 再to one hot 最后threhold
        self.post_label = transforms.Compose(
            [transforms.EnsureType(), transforms.AsDiscrete(to_onehot=self.config['classes'])])  # 后处理标签
        # 在tensorboard画图不需要独热编码
        self.draw = transforms.Compose(
            [transforms.EnsureType(), transforms.Activations(softmax=True), transforms.AsDiscrete(argmax=True)])

        # 在多分类的情况下 需要两个指标 一个算类的平均dice 一个算每一个类的dice
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)  # 不算背景求dice
        self.dice_metric_class = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    def training_step(self, batch, batch_idx):
        image, label = batch['image'], batch['label']
        result = self.model(image)
        loss = self.loss_fn(result, label)
        self.log('loss', loss)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        sw_batch = 4
        result = sliding_window_inference(
            images, self.roi_size, sw_batch, self.model, mode='gaussian'
        )
        loss = self.loss_fn(result, labels)
        outputs_for_dice = [self.post_pred(i) for i in decollate_batch(result)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs_for_dice, y=labels)
        self.dice_metric_class(y_pred=outputs_for_dice, y=labels)
        outputs = [self.draw(i) for i in decollate_batch(result)]

        self.log('val_loss', loss)

        plot_2d_or_3d_image(tag="result", data=outputs, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=images, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        return {'val_loss': loss, 'val_number': len(outputs)}

    # def training_epoch_end(self, step_outputs):
    #     loss = 0.
    #     for result in step_outputs:
    #         loss += result['loss']
    #     loss = loss / len(step_outputs)

    def validation_epoch_end(self, step_outputs):
        val_loss, num_items = 0, 0
        for output in step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        dice_mertric = self.dice_metric
        dice_mertric_class = self.dice_metric_class
        mean_val_dice = dice_mertric.aggregate().item()
        dice_mertric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        class_val_dice = dice_mertric_class.aggregate()
        # channel_dice = [i.item() for i in class_val_dice]
        dice_mertric.reset()

        self.print('\nmean_val_dice: {}\tmean_val_loss: {}\n'.format(mean_val_dice, mean_val_loss))
        self.log('mean_val_loss', mean_val_loss)
        self.log('mean_val_dice', mean_val_dice)

    def configure_optimizers(self):
        # scheduler 在鞍点的时候减少学习率
        # optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        optimizer = Novograd(self.model.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10,
                                                               min_lr=1e-6,
                                                               verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': "loss"
        }
