"""CW-BASS v2 codebase: entry point for training and evaluation.

Supports three --method choices selectable per run:
  cwbass         - original IJCNN 2025 dynamic-threshold baseline
  cwbass_v2      - dynamic threshold + held-out calibration + self-adaptive
                   confidence floor (the journal-extension recipe)
  class_adaptive - per-class adaptive thresholds (used for the
                   negative-result appendix; not the proposed method)

A sibling repository at ../pixcon/ contains a separate paper (PixCon) built
on the same DINOv2 backbone but a different pseudo-label regime.
"""

import argparse
import os

# Reduce CUDA memory fragmentation — critical for DINOv2-B at 518x518 on 40GB
# A100s where the activation graph leaves little slack. Set before torch import
# so the allocator picks it up.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='CW-BASS v2: confidence-weighted boundary-aware SSSS '
                    'with held-out calibration + self-adaptive floor.')

    # Basic settings
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes', 'ade20k'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str,
                        choices=['resnet50', 'resnet101',
                                 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                        default='resnet50')
    parser.add_argument('--model', type=str,
                        choices=['deeplabv3plus', 'pspnet', 'deeplabv2', 'dino'],
                        default='deeplabv3plus')

    # DINOv2 fine-tuning (used when --model dino)
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                        help='Base LR for deepest DINOv2 backbone layer (LLRD)')
    parser.add_argument('--decoder-lr', type=float, default=1e-3,
                        help='LR for the decoder/head when fine-tuning DINOv2')
    parser.add_argument('--layer-decay', type=float, default=0.65,
                        help='Layer-wise LR decay factor for DINOv2 backbone')

    # Semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)

    # Checkpoint resume
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    # Method (used by per-class threshold module)
    parser.add_argument('--method', type=str,
                        choices=['class_adaptive', 'cwbass', 'cwbass_v2',
                                 'freematch', 'softmatch'],
                        default='class_adaptive',
                        help='Thresholding method: class_adaptive (per-class adaptive), '
                             'cwbass (original IJCNN 2025 dynamic threshold), '
                             'cwbass_v2 (CW-BASS dynamic threshold + held-out calibration '
                             '+ self-adaptive floor, the journal-extension recipe), '
                             'freematch (FreeMatch self-adaptive global threshold modulated '
                             'per-class by learning status, run by its own definition), '
                             'softmatch (SoftMatch truncated-Gaussian soft confidence '
                             'weighting in place of a hard mask).')
    # Training scheme — selects the entire train_epoch flow.
    parser.add_argument('--scheme', type=str,
                        choices=['cwbass_v2', 'unimatch_v2'], default='cwbass_v2',
                        help='cwbass_v2: per-class adaptive thresholds (our pipeline); '
                             'unimatch_v2: faithful UniMatch V2 recipe (fixed 0.95 threshold, '
                             '2 strong views with complementary channel dropout).')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='AdamW weight decay for the DINOv2 path (official UniMatch V2 = 0.01)')
    parser.add_argument('--retention-quantile', type=float, default=0.0,
                        help='If >0, unimatch_v2 keeps only the top (1-q) most-confident '
                             'pixels per batch (anti-confidence-concentration); lower-bounded '
                             'by --conf-thresh. 0 disables (pure fixed threshold).')
    parser.add_argument('--conf-thresh', type=float, default=0.95,
                        help='Fixed confidence threshold used by the unimatch_v2 scheme')

    # PixCon (pixel-level contrastive auxiliary)
    parser.add_argument('--use-pixcon', action='store_true',
                        help='Enable pixel-level contrastive auxiliary on top of unimatch_v2')
    parser.add_argument('--pixcon-weight', type=float, default=0.1,
                        help='Loss weight for PixCon auxiliary')
    parser.add_argument('--pixcon-dim', type=int, default=256,
                        help='Projection head output dimension')
    parser.add_argument('--pixcon-bank-size', type=int, default=256,
                        help='Per-class memory bank size')
    parser.add_argument('--pixcon-temp', type=float, default=0.1,
                        help='InfoNCE temperature')
    parser.add_argument('--pixcon-max-anchors', type=int, default=1024,
                        help='Max anchors per iteration (for compute control)')
    parser.add_argument('--pixcon-per-class', type=int, default=64,
                        help='Max anchors per class (balanced sampling)')
    parser.add_argument('--pixcon-bank-filter', type=str,
                        choices=['clean', 'conf'], default='clean',
                        help='Anchor/bank admission rule (ablation). clean (default): '
                             'labeled pixels where student prediction == GT label '
                             '(clean-positive filter). conf: ReCo/U2PL-style baseline -- '
                             'admit pixels with max-softmax confidence >= --conf-thresh '
                             'using the predicted class as label (can admit confidently-wrong '
                             'entries). Isolates the value of the clean-positive filter.')

    # Confidence / threshold parameters
    parser.add_argument('--gamma', type=float, default=1.0, help='Confidence exponent for loss weighting (0 disables confidence weighting)')
    parser.add_argument('--boundary-weight', type=float, default=0.5, help='Weight of the Sobel boundary auxiliary (0 disables it; for ablation)')
    parser.add_argument('--beta', type=float, default=0.5, help='Sigmoid steepness for dynamic threshold')
    parser.add_argument('--base-threshold', type=float, default=0.6, help='Base threshold for confidence filtering')
    parser.add_argument('--min-threshold', type=float, default=0.3,
                        help='Lower clamp tau_min on the dynamic/per-class threshold. '
                             'Raising it toward --conf-thresh probes whether the adaptive gap '
                             'survives when the rule is not pinned at its clamp; at '
                             'tau_min=conf_thresh the dynamic rule reduces to strict.')
    parser.add_argument('--max-threshold', type=float, default=0.95,
                        help='Upper clamp tau_max on the dynamic/per-class threshold.')
    parser.add_argument('--freematch-momentum', type=float, default=0.999,
                        help='EMA momentum for the FreeMatch self-adaptive global threshold (--method freematch)')
    parser.add_argument('--softmatch-ema', type=float, default=0.999,
                        help='EMA momentum for SoftMatch confidence mean/variance (--method softmatch)')
    parser.add_argument('--softmatch-n-sigma', type=float, default=2.0,
                        help='Truncated-Gaussian width (in mean/std units) for SoftMatch soft weights')

    # Class-adaptive thresholding parameters
    parser.add_argument('--calib-frac', type=float, default=0.05,
                        help='Fraction of labeled set held out for unbiased per-class noise estimation')
    parser.add_argument('--no-cutmix', action='store_true',
                        help='Ablation: disable CutMix in the image-strong stream')
    parser.add_argument('--no-fp', action='store_true',
                        help='Ablation: disable the feature-perturbation stream')
    parser.add_argument('--rarity-power', type=float, default=1.0,
                        help='Exponent for rarity-scaled coverage penalty lambda_k')
    parser.add_argument('--floor-momentum', type=float, default=0.99,
                        help='EMA momentum for the self-adaptive confidence floor')
    parser.add_argument('--floor-scale', type=float, default=0.95,
                        help='Scale on EMA mean confidence to form the floor (CW-BASS v2)')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup epochs before per-class thresholds')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay for per-class stats')
    parser.add_argument('--lambda-coverage', type=float, default=0.1, help='Coverage penalty in risk functional')
    parser.add_argument('--delta', type=float, default=0.05,
                        help='legacy Hoeffding confidence parameter; unused by the '
                             'deployed per-class rule, which uses the held-out point '
                             'estimate (paper Eq. 5)')
    parser.add_argument('--epsilon-target', type=float, default=0.1, help='Target noise rate for Beta threshold')

    # Logging and reproducibility
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Two-pass parse so YAML config can override argparse defaults while
    # explicit CLI flags still win. First pass finds --config; we then push
    # the YAML values into the parser's defaults; the final parse applies
    # everything in the standard precedence (CLI > YAML > argparse default).
    pre, _ = parser.parse_known_args()
    if pre.config is not None:
        with open(pre.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader) or {}
        cfg = {k.replace('-', '_'): v for k, v in cfg.items()}
        # Only set defaults for keys the parser actually knows about; silently
        # ignore unrecognised YAML keys so configs can carry comments/notes.
        known = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in known})

    args = parser.parse_args()
    return args


def get_num_classes(dataset):
    return {'pascal': 21, 'cityscapes': 19, 'ade20k': 150}[dataset]


if __name__ == '__main__':
    args = parse_args()

    # Dataset-specific defaults. DINOv2 (patch-14) needs crop sizes divisible by
    # 14 and runs fewer epochs since the backbone is already strongly pretrained.
    is_dino = args.model == 'dino' or args.backbone.startswith('dinov2')
    if is_dino:
        dataset_defaults = {
            'pascal':     {'epochs': 40,  'lr': 0.001, 'crop_size': 518},  # 37*14
            'cityscapes': {'epochs': 120, 'lr': 0.004, 'crop_size': 686},  # 49*14
            'ade20k':     {'epochs': 60,  'lr': 0.001, 'crop_size': 518},
        }
    else:
        dataset_defaults = {
            'pascal': {'epochs': 80, 'lr': 0.001, 'crop_size': 321},
            'cityscapes': {'epochs': 240, 'lr': 0.004, 'crop_size': 801},
            'ade20k': {'epochs': 120, 'lr': 0.001, 'crop_size': 512},
        }
    defaults = dataset_defaults.get(args.dataset, {})
    args.epochs = args.epochs or defaults.get('epochs')
    args.lr = args.lr or (defaults.get('lr') / 16 * args.batch_size)
    args.crop_size = args.crop_size or defaults.get('crop_size')
    args.num_classes = get_num_classes(args.dataset)

    if is_dino and args.crop_size % 14 != 0:
        raise ValueError(f'DINOv2 requires --crop-size divisible by 14, got {args.crop_size}')

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from train import train_full
    train_full(args, device)
