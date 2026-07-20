"""DINOv2 ViT backbone for semantic segmentation.

Loads a pretrained DINOv2 ViT via torch.hub and exposes multi-layer patch-token
features as spatial feature maps, matching the interface the segmentation
decoders expect.

DINOv2 uses patch size 14, so input H/W must be divisible by 14. Unlike a CNN,
all transformer blocks operate at the same spatial resolution (H/14, W/14); the
decoder is responsible for building a feature pyramid from the selected layers.
"""

import torch
import torch.nn as nn


# embed_dim and block count per DINOv2 variant
_DINOV2_SPECS = {
    'dinov2_vits14': {'embed_dim': 384, 'depth': 12, 'layers': (2, 5, 8, 11)},
    'dinov2_vitb14': {'embed_dim': 768, 'depth': 12, 'layers': (2, 5, 8, 11)},
    'dinov2_vitl14': {'embed_dim': 1024, 'depth': 24, 'layers': (4, 11, 17, 23)},
    'dinov2_vitg14': {'embed_dim': 1536, 'depth': 40, 'layers': (9, 19, 29, 39)},
}

PATCH_SIZE = 14


class DINOv2Backbone(nn.Module):
    """Wraps a pretrained DINOv2 ViT, returning 4 intermediate feature maps.

    Args:
        name: one of _DINOV2_SPECS keys (default 'dinov2_vitb14')
        pretrained: load pretrained weights from torch.hub (default True)
        out_layers: which block indices to extract; defaults to the variant's
            standard 4-layer selection.

    Forward returns a list of 4 tensors, each [B, embed_dim, H/14, W/14].
    The `channels` attribute (list of 4 ints, all == embed_dim) mirrors the
    ResNet backbone interface so decoders can introspect feature widths.
    """

    def __init__(self, name='dinov2_vitb14', pretrained=True, out_layers=None):
        super().__init__()
        if name not in _DINOV2_SPECS:
            raise ValueError(f'Unknown DINOv2 variant {name!r}; '
                             f'choose from {list(_DINOV2_SPECS)}')
        spec = _DINOV2_SPECS[name]
        self.name = name
        self.embed_dim = spec['embed_dim']
        self.out_layers = tuple(out_layers) if out_layers is not None else spec['layers']
        self.patch_size = PATCH_SIZE
        # Mirror ResNet interface: 4 feature maps, all at embed_dim width.
        self.channels = [self.embed_dim] * len(self.out_layers)

        self.vit = torch.hub.load('facebookresearch/dinov2', name,
                                  pretrained=pretrained)

    def base_forward(self, x):
        """Return intermediate feature maps as a list of [B, C, h, w] tensors.

        h = H // 14, w = W // 14. Uses DINOv2's get_intermediate_layers with
        reshape=True so CLS tokens are dropped and patch tokens are folded back
        into a spatial grid.
        """
        h, w = x.shape[-2:]
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f'DINOv2 requires input divisible by {self.patch_size}, '
                f'got {h}x{w}')

        feats = self.vit.get_intermediate_layers(
            x, n=self.out_layers, reshape=True, return_class_token=False, norm=True)
        return list(feats)

    def forward(self, x):
        return self.base_forward(x)


def dinov2_vitb14(pretrained=True, **kwargs):
    return DINOv2Backbone('dinov2_vitb14', pretrained=pretrained, **kwargs)


def dinov2_vits14(pretrained=True, **kwargs):
    return DINOv2Backbone('dinov2_vits14', pretrained=pretrained, **kwargs)


def dinov2_vitl14(pretrained=True, **kwargs):
    return DINOv2Backbone('dinov2_vitl14', pretrained=pretrained, **kwargs)
