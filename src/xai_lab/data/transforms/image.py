from __future__ import annotations

from dataclasses import dataclass
from torchvision import transforms

# ImageNet normalization is standard when using pretrained ResNet weights.
# Even for grayscale images converted to RGB, this baseline usually works well.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentConfig:
    """
    Mild face-safe augmentation defaults.

    Why mild?
      For facial emotion, over-aggressive crops/warps can remove key cues
      (eyes/mouth) and make both training and XAI explanations messy/noisy.
    """
    crop_scale_min: float = 0.85
    hflip_p: float = 0.5
    rotation_deg: int = 10
    jitter_brightness: float = 0.15
    jitter_contrast: float = 0.15


def build_transforms(input_size: int, train: bool, aug: AugmentConfig) -> transforms.Compose:
    """
    Build preprocessing pipelines.

    train=True:
      Uses augmentation to reduce overfitting + improve robustness.

    train=False:
      Deterministic preprocessing only, so validation/test metrics are stable.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(aug.crop_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(p=aug.hflip_p),

            # RandomApply keeps augmentation optional (less distortion overall).
            transforms.RandomApply([transforms.RandomRotation(aug.rotation_deg)], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=aug.jitter_brightness, contrast=aug.jitter_contrast)
            ], p=0.5),

            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
