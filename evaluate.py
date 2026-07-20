"""Evaluation entry point for trained models."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from core.metrics import SegmentationMetrics


def get_num_classes(dataset):
    return {'pascal': 21, 'cityscapes': 19, 'ade20k': 150}[dataset]


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--model', type=str, default='deeplabv3plus')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--tta', action='store_true', help='Test-time augmentation')
    args = parser.parse_args()

    num_classes = get_num_classes(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, num_classes).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Handle DataParallel keys
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace('module.', '')] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # Dataset
    if args.dataset == 'ade20k':
        from dataset.ade20k import ADE20KDataset
        valset = ADE20KDataset(args.data_root, 'val')
    else:
        valset = SemiDataset(args.dataset, args.data_root, 'val', None)

    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    metrics = SegmentationMetrics(num_classes)

    with torch.no_grad():
        for batch in tqdm(valloader, desc='Evaluating'):
            if len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch
            images = images.to(device)

            if args.tta:
                logits = model(images, tta=True)
            else:
                logits = model(images)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks_np = masks.numpy() if not isinstance(masks, np.ndarray) else masks
            metrics.add_batch(preds, masks_np)

    per_class_iou, miou = metrics.evaluate()
    print(f'\nmIoU: {miou * 100:.2f}%\n')
    print('Per-class IoU:')
    for k in range(num_classes):
        if per_class_iou[k] > 0:
            print(f'  Class {k:3d}: {per_class_iou[k] * 100:.2f}%')


if __name__ == '__main__':
    main()
