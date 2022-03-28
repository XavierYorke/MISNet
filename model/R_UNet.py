from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.utils import set_determinism
from typing import Any, Callable
from torch import nn
import torch
from pytorch_lightning import seed_everything


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


class R_UNet(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            # channels=(32, 64, 128, 256, 512),
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE,
        )
        # 复现种子点
        set_determinism(seed=seed)
        seed_everything(seed=seed)
        # 网络初始化
        self._model.apply(kaiming_normal_init)

    def forward(self, x):
        return self._model(x)


if __name__ == '__main__':
    model = R_UNet(1228)
    print(model)
