"""
Plot delle metriche di training
===============================
Genera, a partire dagli output di train.py:

  1. Un grafico per OGNI metrica (loss, accuracy, f1, auc, precision, recall)
     che mette a confronto la curva di TRAIN e quella di VAL lungo le epoche.
  2. Le confusion matrix del BEST model su train / val / test set.

Uso:
    python plot_metrics.py --exp_dir checkpoints/baseline_gru_base

Legge:
    <exp_dir>/history.json       (curve per epoca)
    <exp_dir>/predictions.json   (label/pred del best model su train/val/test)
Salva i PNG in:
    <exp_dir>/plots/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


METRICS = [
    ("loss",      "Loss"),
    ("acc",       "Accuracy"),
    ("f1",        "F1 Score"),
    ("auc",       "AUC (ROC)"),
    ("precision", "Precision"),
    ("recall",    "Recall"),
]

CLASS_NAMES = ["Non-crossing", "Crossing"]


def plot_metric_curves(history, out_dir):
    """Un PNG per metrica: Train vs Val vs Test lungo le epoche."""
    epochs = [h["epoch"] for h in history]

    for key, title in METRICS:
        train_key = f"train_{key}"
        val_key   = f"val_{key}"
        test_key  = f"test_{key}"

        has_train = train_key in history[0]
        has_val   = val_key in history[0]
        has_test  = test_key in history[0]

        if not has_train:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [h[train_key] for h in history], marker="o", markersize=3,
                 linewidth=1.8, label="Train")

        title_parts = ["Train"]

        if has_val:
            val_vals = [h[val_key] for h in history]
            plt.plot(epochs, val_vals, marker="s", markersize=3,
                     linewidth=1.8, label="Validation")
            best_i = int(np.argmin(val_vals)) if key == "loss" else int(np.argmax(val_vals))
            plt.axvline(epochs[best_i], color="gray", linestyle="--", alpha=0.5, linewidth=1)
            plt.scatter([epochs[best_i]], [val_vals[best_i]], color="red", zorder=5, s=40,
                        label=f"Best val ({val_vals[best_i]:.3f} @ ep {epochs[best_i]})")
            title_parts.append("Validation")

        if has_test:
            test_vals = [h[test_key] for h in history]
            plt.plot(epochs, test_vals, marker="^", markersize=3,
                     linewidth=1.8, label="Test", color="green")
            if not has_val:
                best_i = int(np.argmin(test_vals)) if key == "loss" else int(np.argmax(test_vals))
                plt.axvline(epochs[best_i], color="gray", linestyle="--", alpha=0.5, linewidth=1)
                plt.scatter([epochs[best_i]], [test_vals[best_i]], color="darkgreen", zorder=5, s=40,
                            label=f"Best test ({test_vals[best_i]:.3f} @ ep {epochs[best_i]})")
            title_parts.append("Test")

        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"{title} — {' vs '.join(title_parts)}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = out_dir / f"metric_{key}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  salvato {path}")


def plot_confusion_matrices(predictions, out_dir):
    """Confusion matrix del best model su train/val/test in un'unica figura."""
    splits = [s for s in ("train", "val", "test") if s in predictions]
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4.5))
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        labels = np.array(predictions[split]["labels"])
        preds  = np.array(predictions[split]["preds"])
        cm = confusion_matrix(labels, preds, labels=[0, 1])

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{split.capitalize()} set")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # annota i conteggi (e la percentuale per riga)
        row_sums = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / row_sums[i, 0] if row_sums[i, 0] else 0
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                        ha="center", va="center", color=color, fontsize=11)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrix — Best Model", fontsize=14)
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  salvato {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Cartella esperimento (es. checkpoints/baseline_gru_base)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Curve metriche --
    hist_path = exp_dir / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        print("Plotting curve metriche (train vs val)...")
        plot_metric_curves(history, out_dir)
    else:
        print(f"[!] history.json non trovato in {exp_dir}")

    # -- Confusion matrix --
    pred_path = exp_dir / "predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            predictions = json.load(f)
        print("Plotting confusion matrix (best model)...")
        plot_confusion_matrices(predictions, out_dir)
    else:
        print(f"[!] predictions.json non trovato in {exp_dir}")

    print(f"\nTutti i grafici in: {out_dir}")


if __name__ == "__main__":
    main()
