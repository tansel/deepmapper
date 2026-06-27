"""Held-out evaluation with arrangement-ensembling, for confusion matrices & metrics.

The iterate-N runner (`runner.run`) accumulates *attribution*; this module accumulates
*predictions*. It fixes ONE stratified train/test split, trains the backbone under N
arrangements, and ensembles the softmax over arrangements into a final prediction, the
principled DeepMapper class call. Returns y_true / y_pred / y_proba + a sklearn report,
ready for `plots.py`. Needs torch (the pure core stays torch-free).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import backbones
from .augment import make_train_transform
from .config import DeepMapperConfig
from .transform import permutation, to_images


@dataclass
class EvalResult:
    y_true: list
    y_pred: list
    y_proba: list                       # (n_test, n_classes) ensembled softmax
    classes: List[str]
    accuracy: float
    macro_f1: float
    per_pass_acc: List[float] = field(default_factory=list)
    report: dict = field(default_factory=dict)   # sklearn classification_report (dict)
    n_passes: int = 0


def evaluate(X, y, config: DeepMapperConfig, class_names: Optional[List[str]] = None,
             test_size: float = 0.25, seed: int = 0) -> EvalResult:
    import numpy as np
    import torch
    from sklearn.metrics import (accuracy_score, classification_report, f1_score)
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)
    n, n_features = X.shape
    num_classes = int(y.max()) + 1
    classes = class_names or [str(i) for i in range(num_classes)]

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    aug = make_train_transform(config.augment)

    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    y_te = y[te]
    proba_sum = np.zeros((len(te), num_classes), dtype=np.float64)
    per_pass = []

    for t in range(config.n_passes):
        perm = None if config.index_features else permutation(seed + t, n_features)
        imgs = np.transpose(to_images(X, buffer=config.buffer, perm=perm), (0, 3, 1, 2))
        Xt = torch.tensor(imgs, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.int64)

        model = backbones.build(config.backbone, num_classes, n_features).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(Xt[tr], yt[tr]),
                            batch_size=config.batch_size, shuffle=True)
        model.train()
        for _ in range(config.epochs):
            for xb, yb in loader:
                xb = aug(xb).to(device); yb = yb.to(device)
                opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xt[te].to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        proba_sum += proba
        per_pass.append(float((proba.argmax(1) == y_te).mean()))

    y_pred = proba_sum.argmax(1)
    rep = classification_report(y_te, y_pred, target_names=classes,
                                output_dict=True, zero_division=0)
    return EvalResult(
        y_true=y_te.tolist(), y_pred=y_pred.tolist(),
        y_proba=(proba_sum / config.n_passes).tolist(), classes=classes,
        accuracy=float(accuracy_score(y_te, y_pred)),
        macro_f1=float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        per_pass_acc=per_pass, report=rep, n_passes=config.n_passes)
