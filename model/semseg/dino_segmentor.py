"""DINOv2 segmentation model with a DPT-lite decoder.

The DINOv2 backbone emits 4 patch-token feature maps, all at H/14 resolution.
This decoder resamples them into a {1/4, 1/8, 1/16, 1/32}-style pyramid (relative
to the patch grid), fuses them bottom-up with residual conv blocks, and predicts
per-pixel logits upsampled to the input resolution.

A `feature_dropout` hook on the fused decoder feature implements the UniMatch-style
feature-space perturbation: the same backbone features produce a second prediction
under channel dropout, supervised by the same teacher pseudo-label.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2Backbone


class _ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.act(x))
        out = self.bn1(out)
        out = self.conv2(self.act(out))
        out = self.bn2(out)
        return out + x


class _FeatureFusionBlock(nn.Module):
    """Fuse a higher-level (coarser) feature with a skip feature, then upsample."""

    def __init__(self, channels):
        super().__init__()
        self.rcu_skip = _ResidualConvUnit(channels)
        self.rcu_out = _ResidualConvUnit(channels)

    def forward(self, x, skip=None):
        if skip is not None:
            # Resize x to the skip's grid before fusing — robust to odd patch
            # grids where down/up resampling factors don't divide evenly.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = x + self.rcu_skip(skip)
        x = self.rcu_out(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class DINOv2Segmentor(nn.Module):
    """DINOv2 + DPT-lite decoder.

    Args:
        backbone: variant name (e.g. 'dinov2_vitb14') or a DINOv2Backbone instance
        nclass: number of segmentation classes
        decoder_dim: common decoder width (default 256)
        feature_dropout: channel-dropout prob for the perturbation stream (default 0.5)
        pretrained: load pretrained backbone weights (default True)
    """

    def __init__(self, backbone='dinov2_vitb14', nclass=21, decoder_dim=256,
                 feature_dropout=0.5, pretrained=True):
        super().__init__()
        if isinstance(backbone, str):
            self.backbone = DINOv2Backbone(backbone, pretrained=pretrained)
        else:
            self.backbone = backbone

        in_chs = self.backbone.channels  # list of 4 widths, all embed_dim
        assert len(in_chs) == 4, 'decoder expects 4 backbone feature maps'

        # Project each ViT layer to the common decoder width.
        self.projects = nn.ModuleList([
            nn.Conv2d(c, decoder_dim, 1, bias=False) for c in in_chs
        ])
        # Resample the 4 same-resolution ViT maps into a coarse->fine pyramid.
        # Index 0 = finest (upsample 4x), 3 = coarsest (downsample 2x).
        self.resamples = nn.ModuleList([
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 4, stride=4),   # 4x up
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, stride=2),   # 2x up
            nn.Identity(),                                               # keep
            nn.Conv2d(decoder_dim, decoder_dim, 3, stride=2, padding=1),  # 2x down
        ])
        # Fusion blocks, applied coarse -> fine.
        self.fusions = nn.ModuleList([_FeatureFusionBlock(decoder_dim) for _ in range(4)])

        self.feature_dropout = nn.Dropout2d(p=feature_dropout)
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, nclass, 1),
        )

    def decode_features(self, feats):
        """Run the DPT-lite fusion, returning the fused decoder feature map.

        Args:
            feats: list of 4 [B, C, h, w] backbone feature maps (h=w=H/14).
        Returns:
            fused: [B, decoder_dim, H', W'] decoder feature (before the head).
        """
        # Project + resample into the pyramid.
        pyr = [self.resamples[i](self.projects[i](feats[i])) for i in range(4)]
        # Fuse coarse (3) -> fine (0).
        x = self.fusions[3](pyr[3])
        x = self.fusions[2](x, pyr[2])
        x = self.fusions[1](x, pyr[1])
        x = self.fusions[0](x, pyr[0])
        return x

    def forward(self, x, perturb_feature=False, return_feature=False):
        """Forward pass.

        Args:
            x: [B, 3, H, W] input (H, W divisible by 14)
            perturb_feature: if True, apply channel dropout to the fused decoder
                feature before the head (UniMatch feature-perturbation stream).
            return_feature: if True, also return the fused decoder feature.
        Returns:
            logits [B, nclass, H, W], or (logits, feature) if return_feature.
        """
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)
        fused = self.decode_features(feats)
        if perturb_feature:
            fused = self.feature_dropout(fused)
        logits = self.head(fused)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        if return_feature:
            return logits, fused
        return logits

    def forward_dual(self, x):
        """Image-strong and feature-perturbation streams from ONE backbone forward.

        UniMatch-style: a single backbone+decoder pass produces the strong-stream
        logits; the same fused feature, after channel dropout, produces the
        feature-perturbation logits. Both are supervised by the same teacher
        pseudo-label. This saves the second backbone graph that would otherwise
        OOM at 518x518 batch 8 on a 40GB GPU.

        Returns:
            (logits_strong, logits_fp), both [B, nclass, H, W] at input resolution.
        """
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)
        fused = self.decode_features(feats)
        logits_s = self.head(fused)
        logits_fp = self.head(self.feature_dropout(fused))
        logits_s = F.interpolate(logits_s, size=(h, w), mode='bilinear', align_corners=True)
        logits_fp = F.interpolate(logits_fp, size=(h, w), mode='bilinear', align_corners=True)
        return logits_s, logits_fp

    def forward_unimatch_dual(self, x, p_drop=0.5):
        """UniMatch V2 dual stream: complementary channel dropout, single forward.

        The input x is the concatenation of two strong-augmented views of the same
        unlabeled batch: x = [s1; s2] with shape [2B, 3, H, W]. The backbone+decoder
        runs once over both halves. We then draw a Bernoulli mask M of shape
        [1, C, 1, 1] with probability p_drop=0.5, scale the first half's fused
        feature by 2M and the second half by 2(1-M). This makes the two student
        predictions see *complementary* feature subsets, increasing the diversity
        of the consistency regularisation. (UniMatch V2, Algorithm 1.)

        Returns:
            (logits_s1, logits_s2), each [B, nclass, H, W].
        """
        if x.shape[0] % 2 != 0:
            raise ValueError('forward_unimatch_dual expects batch concat of two views')
        h, w = x.shape[-2:]
        b = x.shape[0] // 2
        feats = self.backbone.base_forward(x)
        fused = self.decode_features(feats)  # [2B, C, h', w']
        # Complementary channel mask on the fused feature.
        c = fused.shape[1]
        mask = (torch.rand(1, c, 1, 1, device=fused.device) < p_drop).float()
        fused_s1 = fused[:b] * (mask * 2.0)
        fused_s2 = fused[b:] * ((1.0 - mask) * 2.0)
        logits_s1 = self.head(fused_s1)
        logits_s2 = self.head(fused_s2)
        logits_s1 = F.interpolate(logits_s1, size=(h, w), mode='bilinear', align_corners=True)
        logits_s2 = F.interpolate(logits_s2, size=(h, w), mode='bilinear', align_corners=True)
        return logits_s1, logits_s2
