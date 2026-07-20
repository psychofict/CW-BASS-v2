"""Optimizer construction for DINOv2 fine-tuning.

Implements layer-wise learning-rate decay (LLRD): earlier transformer blocks get
exponentially smaller learning rates than later blocks, and the decoder head gets
a much larger LR than the backbone. This is the standard recipe for fine-tuning
large self-supervised ViTs (DINOv2, BEiT, MAE) on dense tasks and is what makes
full fine-tuning stable rather than catastrophic.
"""

import torch


def _vit_layer_id(name, num_blocks):
    """Map a DINOv2 parameter name to a depth index in [0, num_blocks+1].

    0 = patch embed / cls / pos tokens (shallowest), num_blocks+1 = final norm.
    Used to assign each parameter its layer-wise decayed LR.
    """
    if 'backbone.vit' not in name:
        return num_blocks + 1  # non-backbone handled separately; safe default
    tail = name.split('backbone.vit.')[-1]
    if tail.startswith('patch_embed') or 'cls_token' in tail or 'pos_embed' in tail \
            or 'mask_token' in tail or 'register_tokens' in tail:
        return 0
    if tail.startswith('blocks'):
        # e.g. "blocks.7.attn.qkv.weight"  (or "blocks.0.7...." in chunked variants)
        parts = tail.split('.')
        # find the first integer after "blocks"
        for p in parts[1:]:
            if p.isdigit():
                return int(p) + 1
        return num_blocks + 1
    return num_blocks + 1  # norm / head-side of backbone


def build_optimizer(model, backbone_lr, decoder_lr, weight_decay=0.05,
                    layer_decay=0.65, betas=(0.9, 0.999)):
    """Build an AdamW optimizer with layer-wise LR decay.

    Args:
        model: a DINOv2Segmentor (has .backbone.vit with .blocks).
        backbone_lr: base LR for the *deepest* backbone layer (e.g. 1e-5).
        decoder_lr: LR for all non-backbone (decoder/head) params (e.g. 1e-3).
        weight_decay: AdamW weight decay (no decay on norms/biases/tokens).
        layer_decay: per-layer decay factor (<1). Shallower layers get
            backbone_lr * layer_decay ** (depth - layer_id).
        betas: AdamW betas.

    Returns:
        torch.optim.AdamW
    """
    vit = model.backbone.vit
    num_blocks = len(vit.blocks)
    max_layer = num_blocks + 1

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = param.ndim == 1 or name.endswith('.bias') \
            or 'cls_token' in name or 'pos_embed' in name \
            or 'register_tokens' in name or 'mask_token' in name
        wd = 0.0 if no_decay else weight_decay

        if 'backbone.vit' in name:
            layer_id = _vit_layer_id(name, num_blocks)
            scale = layer_decay ** (max_layer - layer_id)
            lr = backbone_lr * scale
            key = f'backbone_l{layer_id}_{"nd" if no_decay else "wd"}'
        else:
            lr = decoder_lr
            key = f'decoder_{"nd" if no_decay else "wd"}'

        if key not in param_groups:
            param_groups[key] = {'params': [], 'lr': lr, 'weight_decay': wd}
        param_groups[key]['params'].append(param)

    return torch.optim.AdamW(list(param_groups.values()), betas=betas)


def poly_lr_scale(current_iter, total_iters, power=0.9):
    """Polynomial LR multiplier in [0, 1]. Multiply each group's base LR by this."""
    return (1.0 - current_iter / max(total_iters, 1)) ** power
