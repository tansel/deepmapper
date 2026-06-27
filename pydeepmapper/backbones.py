"""Swappable backbones, CNN <-> ViT <-> autoencoder behind one interface.

The whole point of the revision: the model is a plug-in. ``build(spec, num_classes)``
returns a ``torch.nn.Module`` for any registered ``kind``; the rest of the pipeline
(``runner``, ``attribution``) never names a concrete architecture.

torch / timm are imported lazily inside the builders, so importing this module is
free and the pure core stays torch-independent. If a backbone is requested without
its dependency installed, a clear ``BackboneUnavailable`` is raised.

Registry:
  cnn_small, a tiny 3-conv net. DEFAULT. Small + locality-biased = fast & strong
                here (the hypothesis the benchmark verifies).
  resnet18, torchvision ResNet-18 (legacy DeepMapper parity baseline).
  vit_cct, Compact Convolutional Transformer (top ViT pick for pseudo-images).
  timm:<name>, any timm model (e.g. timm:mobilevit_xxs, timm:convnext_nano).
  conv_vae, convolutional VAE for the latent-space option (encoder used as head).
  linear, flatten + single Linear: logistic regression on the FULL unfiltered
                feature set, run through the same no-filtering + multi-pass + attribution
                pipeline. The head-to-head baseline showing the architecture is incidental.
  mlp, flatten + 1 hidden layer: dense NON-linear baseline with NO convolution,
                isolating "non-linearity" from "convolution/imagification".

Non-neural classifiers (sklearn LogReg-L1 / random forest / gradient boosting) are NOT
registered here: integrated-gradients attribution needs a differentiable model, so they
would require a separate permutation-importance attribution path (future work).
"""
from __future__ import annotations

from typing import Callable, Dict

from .config import BackboneSpec
from .transform import image_side


class BackboneUnavailable(RuntimeError):
    """A backbone was requested but its dependency (torch/timm) isn't installed."""


def _resolve_img_size(spec: BackboneSpec, n_features: int) -> int:
    return spec.img_size or image_side(n_features)


# --- builders (torch imported lazily) ---------------------------------------
def _build_cnn_small(spec: BackboneSpec, num_classes: int, n_features: int):
    try:
        import torch.nn as nn
    except ImportError as e:                       # pragma: no cover
        raise BackboneUnavailable("cnn_small needs torch") from e

    c = spec.in_chans
    width = int(spec.extra.get("width", 16))

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(c, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
                nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(width * 2, num_classes)

        def forward(self, x):
            x = self.features(x).flatten(1)
            return self.head(x)

    return SmallCNN()


def _build_cnn(spec: BackboneSpec, num_classes: int, n_features: int):
    """Mid-size CNN, 4 conv blocks with downsampling. The capacity step between the
    underfitting `cnn_small` and the heavy `resnet18`; the likely "small but sufficient"
    sweet spot for DeepMapper pseudo-images (to verify on real data)."""
    try:
        import torch.nn as nn
    except ImportError as e:                       # pragma: no cover
        raise BackboneUnavailable("cnn needs torch") from e
    c = spec.in_chans
    w = int(spec.extra.get("width", 32))

    def block(i, o, pool):
        layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return layers

    class MidCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                *block(c, w, True), *block(w, w * 2, True),
                *block(w * 2, w * 4, True), *block(w * 4, w * 4, False),
                nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(w * 4, num_classes))

        def forward(self, x):
            return self.head(self.features(x).flatten(1))

    return MidCNN()


def _build_resnet(spec: BackboneSpec, num_classes: int, n_features: int, depth: int = 18,
                  small_stem: bool = False):
    """torchvision ResNet (18/34/50). ``small_stem`` swaps the ImageNet stem
    (7x7 stride-2 conv + maxpool, which downsamples 4x immediately) for a CIFAR/
    keras-resnet-style 3x3 stride-1 stem with NO maxpool, preserves the single-pixel
    gene signal in sparse pseudo-images (the likely TF-vs-PyTorch reproduction gap)."""
    try:
        import torch.nn as nn
        import torchvision
    except ImportError as e:                       # pragma: no cover
        raise BackboneUnavailable("resnet needs torch + torchvision") from e
    ctor = {18: torchvision.models.resnet18, 34: torchvision.models.resnet34,
            50: torchvision.models.resnet50}[depth]
    model = ctor(weights="DEFAULT" if spec.pretrained else None)
    if small_stem:
        model.conv1 = nn.Conv2d(spec.in_chans, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()              # no early 4x downsample
    elif spec.in_chans != 3:
        model.conv1 = nn.Conv2d(spec.in_chans, 64, 7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_resnet18(spec, num_classes, n_features):
    return _build_resnet(spec, num_classes, n_features, depth=18, small_stem=False)


def _build_timm(spec: BackboneSpec, num_classes: int, n_features: int):
    try:
        import timm
    except ImportError as e:                       # pragma: no cover
        raise BackboneUnavailable("timm backbones need `pip install timm`") from e
    name = spec.kind.split(":", 1)[1] if ":" in spec.kind else spec.extra.get("name")
    if not name:
        raise ValueError("timm backbone requires a model name (timm:<name>)")
    return timm.create_model(
        name, pretrained=spec.pretrained, num_classes=num_classes,
        in_chans=spec.in_chans, img_size=_resolve_img_size(spec, n_features),
        **{k: v for k, v in spec.extra.items() if k != "name"})


def _build_vit_cct(spec: BackboneSpec, num_classes: int, n_features: int):
    """Compact Convolutional Transformer, top ViT pick for pseudo-images.

    Prefers the `vit-pytorch` CCT (small-image native); falls back to a timm ViT.
    """
    img = _resolve_img_size(spec, n_features)
    try:
        from vit_pytorch.cct import CCT
    except ImportError:
        # Fall back to a small timm ViT so the 'vit_cct' kind always resolves to
        # *some* transformer; record the substitution on the module.
        fallback = BackboneSpec(kind="timm:vit_tiny_patch16_224", img_size=img,
                                in_chans=spec.in_chans, extra={"patch_size": 4})
        m = _build_timm(fallback, num_classes, n_features)
        m._deepmapper_note = "CCT unavailable; used timm vit_tiny fallback"
        return m
    return CCT(img_size=img, embedding_dim=int(spec.extra.get("embedding_dim", 128)),
               n_conv_layers=1, kernel_size=3, stride=1, padding=1,
               num_layers=int(spec.extra.get("num_layers", 7)),
               num_heads=int(spec.extra.get("num_heads", 4)),
               num_classes=num_classes, n_input_channels=spec.in_chans)


def _build_conv_vae(spec: BackboneSpec, num_classes: int, n_features: int):
    """Convolutional VAE for the latent-space option. Returns the full VAE; the
    runner can use the encoder mean as a classification/clustering head, and the
    reconstruction residual as the anomaly/perturbation signal."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:                       # pragma: no cover
        raise BackboneUnavailable("conv_vae needs torch") from e

    c, img = spec.in_chans, _resolve_img_size(spec, n_features)
    latent = int(spec.extra.get("latent_dim", 32))
    w = int(spec.extra.get("width", 32))

    class ConvVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.img = img
            self._s = (img + 3) // 4                # encoder downsamples by 4 (two stride-2 convs)
            self.enc = nn.Sequential(
                nn.Conv2d(c, w, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(w, w * 2, 3, 2, 1), nn.ReLU(), nn.Flatten())
            self._fdim = w * 2 * self._s * self._s
            self.fc_mu = nn.Linear(self._fdim, latent)
            self.fc_var = nn.Linear(self._fdim, latent)
            self.classifier = nn.Linear(latent, num_classes)
            # decoder mirrors the encoder; enables unsupervised reconstruction (VAE) training
            self.dec_fc = nn.Linear(latent, self._fdim)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(w * 2, w, 3, 2, 1, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(w, c, 3, 2, 1, output_padding=1))

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_var(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        def decode(self, z):
            h = self.dec_fc(z).view(-1, w * 2, self._s, self._s)
            return self.dec(h)[:, :, :self.img, :self.img]   # crop to input size

        def forward(self, x):                       # classification head over the latent mean
            mu, _ = self.encode(x)
            return self.classifier(mu)

    return ConvVAE()


def _build_linear(spec: BackboneSpec, num_classes: int, n_features: int):
    """Flatten the pseudo-image and apply one Linear layer = logistic regression on the
    full feature vector. The seeded permutation is irrelevant under a flatten (every pixel
    gets its own weight), and integrated gradients on a linear model returns exactly the
    per-feature linear importances after the inverse-permutation gather."""
    try:
        import torch.nn as nn
    except ImportError as e:                       # pragma: no cover
        raise BackboneUnavailable("linear needs torch") from e

    class LinearHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))

        def forward(self, x):
            return self.net(x)

    return LinearHead()


def _build_mlp(spec: BackboneSpec, num_classes: int, n_features: int):
    """Flatten + one hidden layer: a dense non-linear classifier with NO convolution.
    Separates the effect of non-linearity from the effect of the convolution/imagification."""
    try:
        import torch.nn as nn
    except ImportError as e:                       # pragma: no cover
        raise BackboneUnavailable("mlp needs torch") from e
    h = int(spec.extra.get("hidden", 256))
    p = float(spec.extra.get("dropout", 0.2))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(h), nn.ReLU(),
                                     nn.Dropout(p), nn.Linear(h, num_classes))

        def forward(self, x):
            return self.net(x)

    return MLP()


_REGISTRY: Dict[str, Callable] = {
    "linear": _build_linear,
    "mlp": _build_mlp,
    "cnn_small": _build_cnn_small,
    "cnn": _build_cnn,
    "resnet18": _build_resnet18,
    "resnet18_s": lambda s, nc, nf: _build_resnet(s, nc, nf, depth=18, small_stem=True),
    "resnet34_s": lambda s, nc, nf: _build_resnet(s, nc, nf, depth=34, small_stem=True),
    "resnet50_s": lambda s, nc, nf: _build_resnet(s, nc, nf, depth=50, small_stem=True),
    "vit_cct": _build_vit_cct,
    "conv_vae": _build_conv_vae,
}


def build(spec: BackboneSpec, num_classes: int, n_features: int):
    """Build the backbone for ``spec``. ``kind`` is a registry key or ``timm:<name>``."""
    if spec.kind.startswith("timm:"):
        return _build_timm(spec, num_classes, n_features)
    if spec.kind not in _REGISTRY:
        raise ValueError(
            f"unknown backbone kind {spec.kind!r}; "
            f"known: {sorted(_REGISTRY)} or 'timm:<name>'")
    return _REGISTRY[spec.kind](spec, num_classes, n_features)


def available() -> list:
    """Registry keys (plus the dynamic ``timm:<name>`` form)."""
    return sorted(_REGISTRY) + ["timm:<name>"]
