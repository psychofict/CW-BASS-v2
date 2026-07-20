import torch
import torch.nn.functional as F


def detect_boundaries(labels, num_classes=None, ignore_index=255):
    """
    Detects boundaries in segmentation masks using Sobel filters.
    Returns a binary boundary mask of shape [B, H, W] (float).

    Args:
        labels: [B, H, W] integer class labels.
        num_classes: K. If None, inferred from labels.max()+1 with ignored
            pixels (>= ignore_index) clipped out first.
        ignore_index: label value to treat as void (clipped to 0 before one-hot).
    """
    safe = labels.clone()
    valid = safe < (num_classes if num_classes is not None else ignore_index)
    safe[~valid] = 0  # any out-of-range pixels (e.g. 255) go to class 0
    if num_classes is None:
        num_classes = int(safe.max().item()) + 1

    one_hot = F.one_hot(safe, num_classes=num_classes).permute(0, 3, 1, 2).float()

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=labels.device, dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=labels.device, dtype=torch.float32)
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)

    edges_x = F.conv2d(one_hot, sobel_x, padding=1, groups=num_classes)
    edges_y = F.conv2d(one_hot, sobel_y, padding=1, groups=num_classes)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    boundary_mask = edges.sum(dim=1) > 0
    return boundary_mask.float()
