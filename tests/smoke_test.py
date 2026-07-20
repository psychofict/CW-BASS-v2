"""End-to-end smoke test for both training methods.

Creates a tiny synthetic dataset on disk and runs both cwbass and
class_adaptive methods for 2 epochs each on CPU to verify the full
pipeline integrates without crashing.
"""

import os
import sys
import shutil
import tempfile
import argparse

import numpy as np
from PIL import Image
import torch

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def create_synthetic_dataset(root, num_images=8, num_classes=21,
                             img_size=65, split_ratio=0.5):
    """Create a minimal Pascal-like dataset for smoke testing."""
    img_dir = os.path.join(root, 'JPEGImages')
    mask_dir = os.path.join(root, 'SegmentationClassAug')
    splits_dir = os.path.join(root, 'splits')
    pseudo_dir = os.path.join(root, 'pseudo_masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(pseudo_dir, exist_ok=True)

    ids = []
    for i in range(num_images):
        name = f'synth_{i:04d}'
        ids.append(name)

        # Random RGB image
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(img_dir, f'{name}.jpg'))

        # Random segmentation mask (P-mode, values 0..num_classes-1, with some 255)
        mask = np.random.randint(0, num_classes, (img_size, img_size), dtype=np.uint8)
        mask[0, 0] = 255  # at least one ignore pixel
        mask_img = Image.fromarray(mask, mode='P')
        mask_img.save(os.path.join(mask_dir, f'{name}.png'))

    # Write split files in the format: JPEGImages/name.jpg SegmentationClassAug/name.png
    n_labeled = max(2, int(num_images * split_ratio))
    labeled_ids = ids[:n_labeled]
    unlabeled_ids = ids[n_labeled:]
    if not unlabeled_ids:
        unlabeled_ids = ids[n_labeled - 1:]  # at least one

    with open(os.path.join(splits_dir, 'labeled.txt'), 'w') as f:
        for name in labeled_ids:
            f.write(f'JPEGImages/{name}.jpg SegmentationClassAug/{name}.png\n')
    with open(os.path.join(splits_dir, 'unlabeled.txt'), 'w') as f:
        for name in unlabeled_ids:
            f.write(f'JPEGImages/{name}.jpg SegmentationClassAug/{name}.png\n')

    # Write val.txt where SemiDataset expects it: dataset/splits/pascal/val.txt
    # (relative to CWD = project root)
    val_split_dir = os.path.join(PROJECT_ROOT, 'dataset', 'splits', 'pascal')
    os.makedirs(val_split_dir, exist_ok=True)
    val_txt = os.path.join(val_split_dir, 'val.txt')
    val_existed = os.path.exists(val_txt)
    if not val_existed:
        # Use the labeled images as val set for the smoke test
        with open(val_txt, 'w') as f:
            for name in labeled_ids:
                f.write(f'JPEGImages/{name}.jpg SegmentationClassAug/{name}.png\n')

    return splits_dir, pseudo_dir, val_txt if not val_existed else None


def build_args(data_root, splits_dir, pseudo_dir, save_path, method,
               num_classes=21, epochs=2, batch_size=2, crop_size=33):
    """Build a minimal args namespace mimicking parse_args() output."""
    args = argparse.Namespace(
        config=None,
        data_root=data_root,
        dataset='pascal',
        batch_size=batch_size,
        lr=0.001,
        epochs=epochs,
        crop_size=crop_size,
        backbone='resnet50',
        model='deeplabv3plus',
        resume_from=None,
        labeled_id_path=os.path.join(splits_dir, 'labeled.txt'),
        unlabeled_id_path=os.path.join(splits_dir, 'unlabeled.txt'),
        pseudo_mask_path=pseudo_dir,
        save_path=save_path,
        reliable_id_path=None,
        resume=None,
        gamma=1.0,
        beta=0.5,
        base_threshold=0.6,
        decay_factor=0.9,
        use_confidence_decay=False,
        method=method,
        warmup_epochs=1,
        ema_decay=0.99,
        lambda_coverage=0.1,
        delta=0.05,
        epsilon_target=0.1,
        use_wandb=False,
        seed=42,
        num_classes=num_classes,
    )
    return args


def smoke_test_class_adaptive(tmpdir):
    """Smoke test the class_adaptive method."""
    print('\n' + '=' * 60)
    print('SMOKE TEST: class_adaptive method (2 epochs, CPU)')
    print('=' * 60)

    data_root = os.path.join(tmpdir, 'data')
    save_path = os.path.join(tmpdir, 'output_adaptive')
    os.makedirs(save_path, exist_ok=True)

    splits_dir, pseudo_dir, val_txt_created = create_synthetic_dataset(
        data_root, num_images=8)
    args = build_args(data_root, splits_dir, pseudo_dir, save_path,
                      method='class_adaptive')

    device = torch.device('cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        from train import train_full
        train_full(args, device)

        # Check outputs exist
        assert os.path.isdir(save_path), 'Save path not created'
        print('\nclass_adaptive smoke test PASSED')
    finally:
        # Clean up val.txt we created
        if val_txt_created and os.path.exists(val_txt_created):
            os.remove(val_txt_created)


def smoke_test_cwbass(tmpdir):
    """Smoke test the cwbass (baseline) method."""
    print('\n' + '=' * 60)
    print('SMOKE TEST: cwbass method (2 epochs, CPU)')
    print('=' * 60)

    data_root = os.path.join(tmpdir, 'data_cw')
    save_path = os.path.join(tmpdir, 'output_cwbass')
    os.makedirs(save_path, exist_ok=True)

    splits_dir, pseudo_dir, val_txt_created = create_synthetic_dataset(
        data_root, num_images=8)
    args = build_args(data_root, splits_dir, pseudo_dir, save_path,
                      method='cwbass')

    device = torch.device('cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        from copy import deepcopy
        from torch.nn import CrossEntropyLoss
        from torch.utils.data import DataLoader
        from dataset.semi import SemiDataset
        from main import init_basic_elems, train_with_confidence_weighted_learning
        from main import validate_and_checkpoint

        os.makedirs(args.save_path, exist_ok=True)
        criterion = CrossEntropyLoss(ignore_index=255)

        trainset = SemiDataset(args.dataset, args.data_root, 'train',
                               args.crop_size, args.labeled_id_path)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0, drop_last=True)
        valset = SemiDataset(args.dataset, args.data_root, 'val', None)
        valloader = DataLoader(valset, batch_size=1, shuffle=False,
                               num_workers=0, drop_last=False)

        model, optimizer = init_basic_elems(args, device)
        best_miou = 0.0

        for epoch in range(args.epochs):
            print(f'\nEpoch {epoch + 1}/{args.epochs}')
            train_with_confidence_weighted_learning(
                model, deepcopy(model), trainloader, optimizer, criterion,
                gamma=args.gamma, decay_factor=args.decay_factor,
                base_threshold=args.base_threshold, beta=args.beta,
                device=device, use_confidence_decay=args.use_confidence_decay,
            )
            best_miou = validate_and_checkpoint(
                model, valloader, criterion, optimizer, epoch, best_miou,
                args, device,
            )

        print('\ncwbass smoke test PASSED')
    finally:
        if val_txt_created and os.path.exists(val_txt_created):
            os.remove(val_txt_created)


def main():
    tmpdir = tempfile.mkdtemp(prefix='cwbass_smoke_')
    print(f'Temp dir: {tmpdir}')
    try:
        smoke_test_class_adaptive(tmpdir)
        smoke_test_cwbass(tmpdir)
        print('\n' + '=' * 60)
        print('ALL SMOKE TESTS PASSED')
        print('=' * 60)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
