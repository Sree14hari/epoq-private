# python_backend/augmentation_builder.py

from torchvision import transforms


def build_transforms(aug_config: dict, image_size: int = 224):
    """
    Builds train and validation transforms dynamically
    based on augmentation configuration.
    """

    train_transforms = []

    # ----- AUGMENTATIONS -----

    if aug_config.get("horizontalFlip"):
        train_transforms.append(transforms.RandomHorizontalFlip())

    if aug_config.get("verticalFlip"):
        train_transforms.append(transforms.RandomVerticalFlip())

    rotation_cfg = aug_config.get("rotation", {})
    if rotation_cfg.get("enabled"):
        train_transforms.append(
            transforms.RandomRotation(rotation_cfg.get("degrees", 15))
        )

    color_cfg = aug_config.get("colorJitter", {})
    if color_cfg.get("enabled"):
        train_transforms.append(
            transforms.ColorJitter(
                brightness=color_cfg.get("brightness", 0.2),
                contrast=color_cfg.get("contrast", 0.2),
                saturation=color_cfg.get("saturation", 0.2),
            )
        )

    crop_cfg = aug_config.get("randomResizedCrop", {})
    if crop_cfg.get("enabled"):
        train_transforms.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=(
                    crop_cfg.get("scaleMin", 0.8),
                    crop_cfg.get("scaleMax", 1.0),
                ),
            )
        )
    else:
        # Default resizing if no random crop
        train_transforms.append(transforms.Resize((image_size, image_size)))

    # Always at end
    train_transforms.append(transforms.ToTensor())

    train_transform = transforms.Compose(train_transforms)

    # ---- VALIDATION TRANSFORM (NO AUGMENTATION) ----

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    return train_transform, val_transform