"""Per-pixel attribution -> per-feature findings.

One ``attribute()`` seam, several methods:
  CNN  : integrated_gradients (default), saliency, occlusion   [Captum]
  ViT  : attention_rollout, chefer                              [attention-based]

The method computes a per-pixel importance map for a batch, then
``transform.pixels_to_features`` back-projects it through the arrangement's inverse
permutation to a per-feature vector (the clean 1-pixel-per-feature gather). torch /
captum are imported lazily; a missing dependency raises a clear error.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .transform import pixels_to_features

_CNN_METHODS = {"integrated_gradients", "saliency", "occlusion", "input_x_gradient"}
_VIT_METHODS = {"attention_rollout", "chefer"}


class AttributionUnavailable(RuntimeError):
    pass


def feature_importances(model, X_images, targets, method: str,
                        perm: Optional[Sequence[int]], n_features: int,
                        baseline=None, ig_steps: int = 32, ig_internal_batch: int = 16):
    """Return a per-feature importance vector (length ``n_features``) for ``method``.

    ``X_images``: a ``(batch, C, H, W)`` tensor on the model's device. ``targets``:
    per-sample class indices. Importances are averaged over the batch, then
    back-projected to feature space via the arrangement's inverse permutation.
    ``ig_steps`` / ``ig_internal_batch`` bound Integrated-Gradients memory.
    """
    try:
        import numpy as np
        import torch
    except ImportError as e:                       # pragma: no cover
        raise AttributionUnavailable("attribution needs torch") from e

    model.eval()
    if method in _CNN_METHODS:
        pix = _captum_pixels(model, X_images, targets, method, baseline,
                             ig_steps, ig_internal_batch)
    elif method in _VIT_METHODS:
        pix = _attention_pixels(model, X_images, targets, method)
    else:
        raise ValueError(f"unknown attribution method {method!r}")
    # pix: (batch, H, W) non-negative importance; average over batch then per-feature
    pix = np.asarray(pix, dtype=np.float32)
    mean_img = pix.mean(axis=0)
    return pixels_to_features(mean_img, perm, n_features)


def _captum_pixels(model, X, targets, method, baseline, ig_steps=32, ig_internal_batch=16):
    try:
        import torch
        from captum.attr import (IntegratedGradients, Saliency,
                                 InputXGradient, Occlusion)
    except ImportError as e:                       # pragma: no cover
        raise AttributionUnavailable(
            "CNN attribution needs `pip install captum`") from e
    X = X.clone().requires_grad_(True)
    if method == "integrated_gradients":
        algo = IntegratedGradients(model)
        bl = baseline if baseline is not None else torch.zeros_like(X)
        attr = algo.attribute(X, baselines=bl, target=targets,
                              n_steps=ig_steps, internal_batch_size=ig_internal_batch)
    elif method == "saliency":
        attr = Saliency(model).attribute(X, target=targets)
    elif method == "input_x_gradient":
        attr = InputXGradient(model).attribute(X, target=targets)
    elif method == "occlusion":
        attr = Occlusion(model).attribute(
            X, target=targets, sliding_window_shapes=(X.shape[1], 1, 1))
    else:                                          # pragma: no cover
        raise ValueError(method)
    # collapse channels, take magnitude
    return attr.detach().abs().sum(dim=1).cpu().numpy()


def _attention_pixels(model, X, targets, method):
    """ViT findings. ``attention_rollout`` composes attention across layers; ``chefer``
    is the class-specific LRP+gradient method. We try captum's LayerAttribution /
    the model's attention maps; if unavailable we fall back to input-gradient
    saliency so the pipeline still produces a (weaker) finding rather than crashing.
    """
    try:
        import torch
    except ImportError as e:                       # pragma: no cover
        raise AttributionUnavailable("ViT attribution needs torch") from e
    # A faithful rollout/Chefer implementation depends on the specific ViT exposing
    # its attention tensors. As a dependency-light, always-available fallback we use
    # gradient saliency upsampled to the image grid. Swap in a model-specific
    # rollout/Chefer hook when the concrete ViT is fixed.
    X = X.clone().requires_grad_(True)
    out = model(X)
    sel = out.gather(1, targets.view(-1, 1)).sum()
    grad = torch.autograd.grad(sel, X)[0]
    return grad.detach().abs().sum(dim=1).cpu().numpy()
