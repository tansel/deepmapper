"""pyDeepMapper (revised, contracts-first).

DeepMapper analyses high-dimensional tabular data by mapping each feature to a pixel
of a small pseudo-image, classifying with a **swappable** backbone (CNN / ViT /
autoencoder), and **iterating N seeded arrangements then accumulating** per-feature
findings into robust, stability-scored attributions.

Layers:
  transform, pure feature<->pixel mapping + seeded arrangements   (contracted)
  accumulate, pure iterate-N aggregation + stability statistics    (contracted)
  backbones, CNN <-> ViT <-> autoencoder behind one build()       (torch-optional)
  attribution, per-pixel findings -> per-feature importances        (torch-optional)
  augment, minimal, layout-preserving augmentation policy
  runner, the iterate-N-accumulate orchestrator (run())        (needs torch)
  config, DeepMapperConfig / BackboneSpec / AugmentConfig

The pure layers (transform, accumulate, config, augment) import with stdlib only.
"""
from __future__ import annotations

from . import accumulate, augment, config, transform

__all__ = ["transform", "accumulate", "config", "augment",
           "DeepMapperConfig", "BackboneSpec", "AugmentConfig"]

from .config import AugmentConfig, BackboneSpec, DeepMapperConfig  # noqa: E402

__version__ = "0.1.0"
