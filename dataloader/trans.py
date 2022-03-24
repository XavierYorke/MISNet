from monai import transforms


train_transforms = transforms.Compose(
    [transforms.LoadImaged(keys=["image", "label"]),
     transforms.ScaleIntensityRanged(
         keys=["image"], a_min=-325, a_max=325,
         b_min=0.0, b_max=1.0, clip=True,
     ),
     transforms.AddChanneld(keys=["image", "label"]),
     transforms.SpatialPadd(keys=["image", "label"],
                            spatial_size=[512, 512, 96],
                            method='end'),
     transforms.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                       spatial_size=[96, 96, 96],
                                       num_samples=2, pos=10, neg=1),
     transforms.EnsureTyped(keys=["image", "label"])
     ]

)

val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.ScaleIntensityRanged(
        keys=["image"], a_min=-325, a_max=325,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    transforms.AddChanneld(keys=["image", "label"]),
    transforms.SpatialPadd(keys=["image", "label"],
                           spatial_size=[512, 512, 96],
                           method='end'),
    transforms.RandCropByPosNegLabeld(keys=["image", "label"],
                                      label_key="label",
                                      spatial_size=[96, 96, 96],
                                      num_samples=2, pos=10, neg=1),
    transforms.EnsureTyped(keys=["image", "label"])
])
