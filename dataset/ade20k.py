"""ADE20K dataset for semi-supervised semantic segmentation.

150 classes, 20210 train / 2000 val images.
Extreme class imbalance makes it ideal for showcasing per-class thresholds.
"""

import math
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import crop, hflip, normalize, resize, blur, cutout


ADE20K_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror',
    'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock',
    'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base',
    'box', 'column', 'signboard', 'chest of drawers', 'counter',
    'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
    'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
    'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind',
    'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench',
    'countertop', 'stove', 'palm', 'kitchen island', 'computer',
    'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus',
    'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
    'streetlight', 'booth', 'television', 'airplane', 'dirt track',
    'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman',
    'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
    'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool',
    'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike',
    'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
    'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase',
    'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag',
]


class ADE20KDataset(Dataset):
    """ADE20K dataset for semi-supervised segmentation.

    Args:
        root: root directory containing ADEChallengeData2016/
        mode: 'train', 'val', 'semi_train', or 'label'
        size: crop size for training
        labeled_id_path: path to labeled image IDs
        unlabeled_id_path: path to unlabeled image IDs
        pseudo_mask_path: path to pseudo-label masks
    """

    NUM_CLASSES = 150

    def __init__(self, root, mode, size=None, labeled_id_path=None,
                 unlabeled_id_path=None, pseudo_mask_path=None):
        self.root = root
        self.mode = mode
        self.size = size
        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = (
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids))
                + self.unlabeled_ids
            )
        elif mode == 'val':
            img_dir = os.path.join(root, 'images', 'validation')
            ann_dir = os.path.join(root, 'annotations', 'validation')
            self.ids = self._build_val_ids(img_dir, ann_dir)
        elif mode == 'train':
            with open(labeled_id_path, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'label':
            with open(unlabeled_id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def _build_val_ids(self, img_dir, ann_dir):
        """Build list of (image_path, annotation_path) strings for val."""
        ids = []
        if not os.path.isdir(img_dir):
            return ids
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith('.jpg'):
                ann_fname = fname.replace('.jpg', '.png')
                img_path = os.path.join('images', 'validation', fname)
                ann_path = os.path.join('annotations', 'validation', ann_fname)
                ids.append(f'{img_path} {ann_path}')
        return ids

    def __getitem__(self, item):
        id_str = self.ids[item]
        parts = id_str.split(' ')
        img_path = os.path.join(self.root, parts[0])
        img = Image.open(img_path).convert('RGB')

        if self.mode == 'val' or self.mode == 'label':
            mask_path = os.path.join(self.root, parts[1])
            mask = Image.open(mask_path)
            img, mask = normalize(img, mask)
            # ADE20K annotations: 0 = background mapped to class, but 0 is actually
            # "wall" in the standard indexing. Labels are 1-indexed in the raw data;
            # subtract 1 so classes are 0-149, with 0 (originally unlabeled) → 255
            mask[mask == 0] = 256  # temp
            mask = mask - 1
            mask[mask == 255] = 255  # originally 0 → 255 (ignore)
            return img, mask, id_str

        # Training modes
        if self.mode == 'train' or (self.mode == 'semi_train' and id_str in self.labeled_ids):
            mask_path = os.path.join(self.root, parts[1])
            mask = Image.open(mask_path)
        else:
            fname = os.path.basename(parts[1]) if len(parts) > 1 else os.path.basename(parts[0]).replace('.jpg', '.png')
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # Augmentations
        base_size = 512
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # Strong augmentation on unlabeled
        if self.mode == 'semi_train' and id_str in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)

        img, mask = normalize(img, mask)
        # Fix ADE20K indexing
        mask[mask == 0] = 256
        mask = mask - 1
        mask[mask == 255] = 255
        return img, mask

    def __len__(self):
        return len(self.ids)
