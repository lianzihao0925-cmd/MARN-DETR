"""Opt-in original COCO annotation identities; official transforms/collation stay in use."""
from rfdetr.datasets.coco import CocoDetection, ConvertCoco
from rfdetr.training.module_data import RFDETRDataModule


def enable_object_ids(dataset):
    if not isinstance(dataset, CocoDetection) or not isinstance(dataset.prepare, ConvertCoco):
        raise TypeError("C3A requires the official CocoDetection / ConvertCoco path")
    if dataset.prepare.cat2label is not None:
        raise ValueError("C3A expects original COCO category IDs, not remapped labels")
    dataset.prepare.marn_keep_object_ids = True
    return dataset


class MARNDataModule(RFDETRDataModule):
    """Current scope: ordinary COCO detection, CPU augmentation, unpadded targets.

    IDs are enabled only on the training dataset, before workers are created.
    The official validation dataset and all loader/collation methods are inherited.
    """

    def __init__(self, model_config, train_config):
        backend = getattr(train_config.augmentation_backend, "value", train_config.augmentation_backend)
        if train_config.dataset_file != "coco":
            raise ValueError("C3A supports dataset_file='coco' only")
        if backend not in {"cpu", "torchvision", "albumentations"}:
            raise ValueError("C3A requires an explicit CPU augmentation backend")
        if train_config.pad_targets_to is not None:
            raise ValueError("C3A has not validated padded target IDs; use pad_targets_to=None")
        if model_config.segmentation_head or model_config.use_grouppose_keypoints:
            raise ValueError("C3A currently supports ordinary detection only")
        super().__init__(model_config, train_config)

    def setup(self, stage):
        super().setup(stage)
        if self._dataset_train is not None:
            enable_object_ids(self._dataset_train)
