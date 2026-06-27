"""The DeepMapper iterate-N-accumulate orchestrator.

Generalises the legacy ``deepmap(X, y, num_passes, min_accuracy, max_tries)`` loop:
each pass draws a fresh feature->pixel **arrangement** (seeded permutation), builds
pseudo-images, trains the **swappable backbone**, gates on held-out accuracy, then
attributes and back-projects to per-feature findings. After N accepted passes the
pure accumulators (``accumulate.py``) produce the robust, stability-scored result.

The numeric heavy-lifting needs torch; the *accumulation* is the pure, contracted
core and is unit-tested without torch. Importing this module is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import accumulate as acc
from . import attribution, backbones
from .augment import make_train_transform
from .config import DeepMapperConfig
from .earlystop import EarlyStopper
from .transform import permutation, to_images


@dataclass
class Findings:
    """Accumulated per-feature result of a run."""
    n_features: int
    n_passes_run: int
    n_accepted: int
    accuracies: List[float] = field(default_factory=list)
    stop_epochs: List[int] = field(default_factory=list)
    median_importance: List[float] = field(default_factory=list)
    mean_rank: List[float] = field(default_factory=list)
    selection_frequency: List[float] = field(default_factory=list)
    feature_names: Optional[List[str]] = None

    def ranking(self, top: int = 20):
        """Top features by selection frequency (then median importance), as
        ``(name_or_index, selection_frequency, median_importance)`` tuples."""
        idx = sorted(range(self.n_features),
                     key=lambda i: (-self.selection_frequency[i], -self.median_importance[i]))
        out = []
        for i in idx[:top]:
            name = self.feature_names[i] if self.feature_names else i
            out.append((name, self.selection_frequency[i], self.median_importance[i]))
        return out


def run(X, y, config: DeepMapperConfig, feature_names: Optional[List[str]] = None) -> Findings:
    """Execute the full iterate-N-accumulate pipeline. Returns :class:`Findings`.

    ``X``: ``(n_samples, n_features)`` array-like. ``y``: integer class labels.
    """
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:                       # pragma: no cover
        raise RuntimeError(
            "pydeepmapper.runner.run needs torch + numpy; the pure accumulation "
            "API (pydeepmapper.accumulate) works without them.") from e

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)
    n_samples, n_features = X.shape
    num_classes = int(y.max()) + 1
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"                              # Apple Silicon GPU
    else:
        device = "cpu"
    # Captum gradient attribution is most reliable off-MPS; attribute on CPU when
    # training on MPS (small capped batch, so the device hop is cheap).
    attr_device = "cpu" if device == "mps" else device
    aug = make_train_transform(config.augment)

    importance_lists: List[List[float]] = []
    rank_lists: List[List[float]] = []
    top_sets: List[List[int]] = []
    accuracies: List[float] = []
    stop_epochs: List[int] = []
    accepted = 0
    passes_run = 0

    for t in range(config.effective_max_tries()):
        if accepted >= config.n_passes:
            break
        passes_run += 1
        seed = config.seed + t
        perm = None if config.index_features else permutation(seed, n_features)

        images = to_images(X, buffer=config.buffer, perm=perm)         # (N, dim, dim, 1)
        images = np.transpose(images, (0, 3, 1, 2))                    # -> (N, C, H, W)
        Xt = torch.tensor(images, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.int64)

        n_test = max(1, int(round(config.test_size * n_samples)))
        g = torch.Generator().manual_seed(seed)
        perm_idx = torch.randperm(n_samples, generator=g)
        te, tr = perm_idx[:n_test], perm_idx[n_test:]
        # carve a validation slice from train to monitor the plateau; te stays
        # untouched for the reported accuracy and for attribution.
        n_val = max(1, int(round(config.val_size * len(tr)))) if config.early_stop else 0
        val, trf = (tr[:n_val], tr[n_val:]) if n_val else (tr[:0], tr)

        model = backbones.build(config.backbone, num_classes, n_features).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(Xt[trf], yt[trf]),
                            batch_size=config.batch_size, shuffle=True)

        # train to the held-out-loss plateau (Keras EarlyStopping), keep best weights
        stopper = EarlyStopper(config.patience, config.min_delta, config.min_epochs, mode="min")
        best_state = None
        stop_epoch = config.epochs
        model.train()
        for epoch in range(config.epochs):
            for xb, yb in loader:
                xb = aug(xb).to(device)
                yb = yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            if not config.early_stop or n_val == 0:
                continue
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(Xt[val].to(device)), yt[val].to(device)))
            model.train()
            stop = stopper.step(epoch, vloss)
            if stopper.improved:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if stop:
                stop_epoch = epoch + 1
                break
        if config.early_stop and best_state is not None:
            model.load_state_dict(best_state)
            stop_epoch = stopper.best_epoch + 1
        stop_epochs.append(stop_epoch)

        model.eval()
        with torch.no_grad():
            pred = model(Xt[te].to(device)).argmax(1).cpu()
        acc_t = float((pred == yt[te]).float().mean())
        accuracies.append(acc_t)

        if not acc.accept_pass(acc_t, config.min_accuracy):
            continue                                                  # pass rejected
        accepted += 1

        # cap the number of test samples fed to attribution (bounds memory/time)
        attr_idx = te[:config.attribution_samples]
        model.to(attr_device)
        imp = attribution.feature_importances(
            model, Xt[attr_idx].to(attr_device), yt[attr_idx].to(attr_device),
            config.attribution, perm, n_features,
            ig_steps=config.ig_steps, ig_internal_batch=config.ig_internal_batch)
        imp = [float(v) for v in imp]
        importance_lists.append(imp)
        rank_lists.append([float(r) for r in acc.ranks_from_importances(imp)])
        top_sets.append(acc.top_k_set(imp, config.top_k))

    if accepted == 0:
        return Findings(n_features=n_features, n_passes_run=passes_run, n_accepted=0,
                        accuracies=accuracies, stop_epochs=stop_epochs,
                        feature_names=feature_names)

    return Findings(
        n_features=n_features, n_passes_run=passes_run, n_accepted=accepted,
        accuracies=accuracies, stop_epochs=stop_epochs,
        median_importance=acc.median_importance(importance_lists),
        mean_rank=acc.mean_rank(rank_lists),
        selection_frequency=acc.selection_frequency(top_sets, n_features),
        feature_names=feature_names)
