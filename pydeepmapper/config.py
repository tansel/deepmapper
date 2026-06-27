"""Configuration for a DeepMapper run, plain dataclasses, no heavy deps.

Importing this module is cheap (stdlib only); torch/timm are only touched when a
backbone is actually built. The defaults encode two project decisions:

* **Minimal augmentation.** Pseudo-images are simpler and far less variable than
  natural photos, and geometric augmentations (flips/rotations/crops) would scramble
  the fixed feature->pixel layout. So augmentation is OFF by default, see
  ``docs/design.md`` and the benchmark that verifies this.
* **Small backbone first.** The default backbone is a small CNN; the hypothesis
  that small, locality-biased models are faster *and* better here than large
  data-hungry ones is something the benchmark is set up to verify.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AugmentConfig:
    """Augmentation policy. Default = identity (no augmentation)."""
    enabled: bool = False
    # Only signal-preserving options are offered; geometric transforms are
    # intentionally absent because they break the fixed feature->pixel layout.
    gaussian_noise_std: float = 0.0     # SmoothGrad-style input jitter (0 = off)


@dataclass
class BackboneSpec:
    """Which model sits behind the pseudo-image. CNN <-> ViT <-> autoencoder swap
    happens here and nowhere else (see backbones.build)."""
    kind: str = "cnn_small"             # cnn_small | resnet18 | vit_cct | timm:<name> | conv_vae
    img_size: int = 0                   # 0 = derive from feature count
    in_chans: int = 1
    pretrained: bool = False            # ImageNet weights don't transfer to pseudo-images
    # free-form extras passed to the builder (e.g. timm kwargs, latent_dim for AE)
    extra: dict = field(default_factory=dict)


@dataclass
class DeepMapperConfig:
    """A full iterate-N-accumulate run."""
    # iterate-N-accumulate
    n_passes: int = 30                  # N arrangements to accumulate over
    min_accuracy: float = 0.0           # per-pass quality gate (held-out accuracy)
    max_tries: int = 0                  # 0 = max_tries == n_passes; else cap retries
    top_k: int = 20                     # top-k for selection-frequency stability stat
    # transform
    buffer: int = 0                     # extra pixels beyond feature count
    index_features: bool = False        # True = fixed (identity) arrangement, no shuffle
    # training
    epochs: int = 20                    # max epochs; early stop ends sooner at a plateau
    batch_size: int = 64
    lr: float = 1e-3
    test_size: float = 0.25
    # plateau early-stopping (Keras EarlyStopping semantics, restore_best_weights)
    early_stop: bool = True             # monitor held-out loss, stop when it plateaus
    patience: int = 20                  # epochs of no improvement before stopping
    min_epochs: int = 40                # floor: never stop in the undertrained zone
    min_delta: float = 1e-3             # smallest loss drop that counts as improvement
    val_size: float = 0.15              # fraction of the train split held out to monitor
    # attribution
    attribution: str = "integrated_gradients"   # integrated_gradients | saliency | occlusion | attention_rollout | chefer
    attribution_samples: int = 64               # cap test samples fed to attribution (memory)
    ig_steps: int = 32                          # Integrated Gradients path steps
    ig_internal_batch: int = 16                 # IG internal batch size (bounds memory)
    # composition
    backbone: BackboneSpec = field(default_factory=BackboneSpec)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    seed: int = 0                       # base seed; pass t uses seed + t

    def effective_max_tries(self) -> int:
        return self.max_tries if self.max_tries > 0 else self.n_passes
