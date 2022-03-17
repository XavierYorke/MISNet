import glob
import os
from monai.data import list_data_collate, ThreadDataLoader, Dataset


class MISDataset:
    def __init__(self, data_dir, batch_size, train_trans, val_trans):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_trans = train_trans
        self.val_trans = val_trans

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(
            glob.glob(os.path.join(self.data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(self.data_dir, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        # 训练集 测试集 验证机集 划分
        train_files, val_files, test_files = data_dicts[:-20], data_dicts[-20:], data_dicts[-20:]
        # define the data transforms
        train_transforms = self.train_trans
        val_transforms = self.val_trans

        train_transforms.set_random_state(seed=0)
        val_transforms.set_random_state(seed=0)

        self.train_ds = Dataset(
            data=train_files, transform=train_transforms,
        )

        self.val_ds = Dataset(
            data=val_files, transform=val_transforms

        )

        self.test_ds = Dataset(
            data=test_files, transform=val_transforms

        )

    # 定义相关loader 为了异步加载使用 ThreadDataLoader
    def train_dataloader(self):
        train_loader = ThreadDataLoader(self.train_ds, buffer_size=12, num_workers=6,
                                        batch_size=self.batch_size,
                                        shuffle=True, collate_fn=list_data_collate)
        return train_loader

    def val_dataloader(self):
        val_loader = ThreadDataLoader(
            self.val_ds, batch_size=1, num_workers=24)
        return val_loader

    def test_dataloader(self):
        test_loader = ThreadDataLoader(
            self.test_ds, batch_size=1, num_workers=24)
        return test_loader
