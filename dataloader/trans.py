from monai import transforms

train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.ScaleIntensityRanged(
        keys=["image"], a_min=100, a_max=300,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    transforms.AddChanneld(keys=["image", "label"]),
    # transforms.SpatialPadd(keys=["image", "label"],
    #                        spatial_size=[512, 512, 128],
    #                        method='symmetric'),
    # transforms.RandZoomd(keys=['image', 'label'], min_zoom=0.9, max_zoom=1.1, prob=0.2),
    transforms.RandCropByPosNegLabeld(keys=["image", "label"],
                                      label_key="label",
                                      spatial_size=[64, 64, 64],
                                      num_samples=2, pos=1, neg=1),
    # transforms.RandRotated(keys=['image', 'label'], range_x=0.4, range_y=0.4, range_z=0.4, keep_size=True, prob=0.2),
    # transforms.RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.2),
    transforms.EnsureTyped(keys=["image", "label"])
])

val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.ScaleIntensityRanged(
        keys=["image"], a_min=100, a_max=300,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    transforms.AddChanneld(keys=["image", "label"]),
    # transforms.SpatialPadd(keys=["image", "label"],
    #                        spatial_size=[512, 512, 128],
    #                        method='symmetric'),
    # transforms.RandCropByPosNegLabeld(keys=["image", "label"],
    #                                   label_key="label",
    #                                   spatial_size=[64, 64, 64],
    #                                   num_samples=2, pos=1, neg=0),
    transforms.EnsureTyped(keys=["image", "label"])
])
