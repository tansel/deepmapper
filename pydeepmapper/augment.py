"""Augmentation policy for pseudo-images, deliberately minimal.

Pseudo-images are simpler and far less variable than natural photos, and the
feature->pixel layout is *fixed and meaningful* (pixel (i,j) is always feature g).
So the photo-vision augmentation toolbox is actively harmful here:

  * flips / rotations / crops / scaling  -> scramble the feature layout (a rotated
    pseudo-image maps features to the wrong genes); NEVER use them.
  * colour jitter / blur                 -> meaningless on a 1-channel intensity map.

The only augmentation that respects the layout is small-amplitude **value noise**
(SmoothGrad-style), which can act as a mild regulariser without moving any feature.
It is OFF by default. ``docs/design.md`` benchmarks aug-vs-no-aug to confirm that,
in this regime, no augmentation is as good or better (the project hypothesis).
"""
from __future__ import annotations

from .config import AugmentConfig


def make_train_transform(cfg: AugmentConfig):
    """Return a callable applied to a ``(C,H,W)`` tensor at training time.

    Identity unless ``cfg.enabled`` and a layout-preserving option is configured.
    """
    if not cfg.enabled or cfg.gaussian_noise_std <= 0.0:
        return lambda x: x

    std = float(cfg.gaussian_noise_std)

    def _noise(x):
        import torch
        return x + torch.randn_like(x) * std

    return _noise


def describe(cfg: AugmentConfig) -> str:
    if not cfg.enabled:
        return "none (identity), pseudo-images need no photo augmentation"
    parts = []
    if cfg.gaussian_noise_std > 0:
        parts.append(f"gaussian value-noise sigma={cfg.gaussian_noise_std}")
    return ", ".join(parts) or "enabled but no layout-preserving op configured"
