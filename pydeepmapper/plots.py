"""Publication-quality figures for DeepMapper evaluations.

matplotlib only (no seaborn). Every figure is saved as BOTH a 300-DPI PNG and a vector
PDF, with a consistent clean style. Legends are placed OUTSIDE the data area so they never
overlap bars. All functions take an output stem (no extension) and write ``<stem>.png`` +
``<stem>.pdf``.
"""
from __future__ import annotations

from typing import Dict, List

# colourblind-friendly palette (Okabe-Ito based)
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
_GREY = "#9A9A9A"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.axisbelow": True,
    })
    return plt


def _save(fig, stem: str):
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [stem + ".png", stem + ".pdf"]


def confusion_matrix(y_true, y_pred, classes: List[str], stem: str,
                     title: str = "Confusion matrix", normalize: bool = True):
    from sklearn.metrics import confusion_matrix as _cm, ConfusionMatrixDisplay
    plt = _style()

    labels = list(range(len(classes)))
    cm = _cm(y_true, y_pred, labels=labels)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Render with sklearn's ConfusionMatrixDisplay so axis/label placement is the library's
    # trusted logic (display_labels -> true on y / predicted on x; diagonal is true==pred by
    # construction). We only post-decorate: row-normalized recall + raw count per cell.
    fig, ax = plt.subplots(figsize=(1.05 * len(classes) + 2.6, 1.05 * len(classes) + 2.0))
    ax.grid(False)
    disp = ConfusionMatrixDisplay(confusion_matrix=(cmn if normalize else cm),
                                  display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True,
              values_format=".2f" if normalize else "d")
    if normalize:
        disp.im_.set_clim(0, 1.0)
        for i in range(len(classes)):
            for j in range(len(classes)):
                disp.text_[i, j].set_text(f"{cmn[i, j]:.2f}\n({cm[i, j]})")
                disp.text_[i, j].set_fontsize(10)
    disp.im_.colorbar.set_label(
        "fraction of true class (diagonal = recall)" if normalize else "count", fontsize=10)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title, pad=12)
    # rotation_mode="anchor" pins each label under its OWN column (without it the slant
    # drifts the label toward the neighbour, which misreads as a mis-aligned diagonal)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    return _save(fig, stem)


def per_class_metrics(report: Dict, classes: List[str], stem: str,
                      title: str = "Per-class precision / recall / F1"):
    import numpy as np
    plt = _style()
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1")]
    vals = {k: [report[c][k] for c in classes] for k, _ in metrics}
    x = np.arange(len(classes)); w = 0.25
    fig, ax = plt.subplots(figsize=(1.5 * len(classes) + 2.0, 4.6))
    bars = []
    for i, (k, lbl) in enumerate(metrics):
        b = ax.bar(x + (i - 1) * w, vals[k], w, label=lbl,
                   color=_PALETTE[i], edgecolor="white", linewidth=0.5)
        bars.append((b, vals[k]))
    # value labels
    for b, vv in bars:
        for rect, v in zip(b, vv):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylim(0, 1.16); ax.set_ylabel("score")
    ax.set_title(title, pad=12)
    ax.grid(axis="x", visible=False)
    # legend in the top-right corner, anchored just OUTSIDE the axes so it never overlaps bars
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=True,
              framealpha=0.9, edgecolor="#cccccc", handlelength=1.3)
    return _save(fig, stem)


def method_comparison(rows: List[Dict], stem: str, metric_key: str = "accuracy",
                      title: str = "DeepMapper vs scanpy"):
    """rows: [{'method': str, 'accuracy': float, ...}]"""
    plt = _style()
    names = [r["method"] for r in rows]
    vals = [r.get(metric_key, 0.0) for r in rows]
    colors = [_PALETTE[0] if "DeepMapper" in n else _GREY for n in names]
    fig, ax = plt.subplots(figsize=(1.7 * len(rows) + 1.2, 4.6))
    bars = ax.bar(names, vals, width=0.62, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.12); ax.set_ylabel(metric_key.replace("_", " "))
    ax.set_title(title, pad=12); ax.grid(axis="x", visible=False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    return _save(fig, stem)


def top_genes(genes: List[str], scores: List[float], stem: str,
              highlight=None, title: str = "Top attribution genes"):
    plt = _style()
    hl = highlight or (lambda g: False)
    n = len(genes)
    fig, ax = plt.subplots(figsize=(5.8, 0.32 * n + 1.4))
    ax.grid(axis="y", visible=False)
    y = list(range(n))[::-1]
    colors = [_PALETTE[1] if hl(g) else _PALETTE[0] for g in genes]
    ax.barh(y, scores, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(genes, fontsize=8)
    ax.set_xlabel("accumulated importance"); ax.set_title(title, pad=10)
    if any(hl(g) for g in genes):
        from matplotlib.patches import Patch
        # top-right corner, anchored just outside so it never overlaps the (longest, topmost) bars
        ax.legend(handles=[Patch(color=_PALETTE[1], label="ribosomal"),
                           Patch(color=_PALETTE[0], label="other")],
                  loc="upper left", bbox_to_anchor=(1.0, 1.0),
                  frameon=True, framealpha=0.9)
    return _save(fig, stem)
