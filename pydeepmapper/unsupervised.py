"""Unsupervised DeepMapper: imagify, train a convolutional VAE on reconstruction + KL (no labels),
return the latent embedding. The label-free counterpart to the supervised runner, kept as a first-class
provision so the standard-pipeline comparison can be run unsupervised-vs-unsupervised.

    from pydeepmapper.unsupervised import embed_unsupervised
    Z = embed_unsupervised(X, config)        # (n_samples, latent_dim) encoder means

Cluster Z (k-means / Leiden) and score against held-out labels to compare with PCA+clustering, all
without the network ever seeing a label.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from .config import DeepMapperConfig, BackboneSpec
from . import backbones, transform


def embed_unsupervised(X, config: Optional[DeepMapperConfig] = None, beta: float = 1.0,
                       latent_dim: int = 32, seed: int = 0, mode: str = "vae"):
    """Train a conv encoder-decoder on the imagified data with no labels; return embeddings.

    ``mode``: "vae" (reconstruction + beta*KL, sampled latent) or "ae" (plain autoencoder,
    reconstruction only, deterministic latent). ``beta`` weights the KL term (vae only). Returns a
    ``(n_samples, latent_dim)`` numpy array of encoder means.
    """
    import torch
    import torch.nn.functional as F

    config = config or DeepMapperConfig()
    torch.manual_seed(seed)
    Xi = transform.to_images(X)                          # (n, dim, dim, 1)
    Xt = torch.tensor(np.transpose(Xi, (0, 3, 1, 2)), dtype=torch.float32)  # (n, 1, dim, dim)
    n, _, dim, _ = Xt.shape
    n_features = np.asarray(X).shape[1]

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    spec = BackboneSpec(kind="conv_vae", extra={"latent_dim": latent_dim,
                                                "width": config.backbone.extra.get("width", 32)})
    model = backbones.build(spec, num_classes=2, n_features=n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    model.train()
    for _epoch in range(config.epochs):
        rng.shuffle(idx)
        for s in range(0, n, config.batch_size):
            xb = Xt[idx[s:s + config.batch_size]].to(device)
            mu, logvar = model.encode(xb)
            z = mu if mode == "ae" else model.reparameterize(mu, logvar)
            xhat = model.decode(z)
            recon = ((xhat - xb) ** 2).sum(dim=[1, 2, 3]).mean()
            if mode == "ae":
                loss = recon
            else:
                kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
                loss = recon + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        out = []
        for s in range(0, n, 256):
            mu, _ = model.encode(Xt[s:s + 256].to(device))
            out.append(mu.cpu().numpy())
    return np.concatenate(out, 0)
