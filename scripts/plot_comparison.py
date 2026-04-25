from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = ("contrastive_only", "contrastive_ot", "mrgs")
SUMMARY_KEYS = (
    "retrieval_top1",
    "retrieval_top5",
    "retrieval_mean_rank",
    "retrieval_median_rank",
    "row_top1",
    "row_top5",
    "stimulus_top1",
    "stimulus_top5",
    "class_top1",
    "class_top5",
    "rdm_spearman",
    "row_rdm_spearman",
    "stimulus_rdm_spearman",
    "class_rdm_spearman",
)


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def final_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    epochs = metrics.get("epochs", [])
    if epochs:
        return epochs[-1]
    return metrics["initial_eval"]


def write_summary(root: Path, all_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        final = final_epoch(all_metrics[method])
        row = {"method": method}
        row.update({key: final.get(key) for key in SUMMARY_KEYS})
        rows.append(row)

    output = root / "summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *SUMMARY_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_bar(root: Path, rows: list[dict[str, Any]], key: str, ylabel: str) -> None:
    plt = import_pyplot()
    methods = [row["method"] for row in rows]
    values = [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, values)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("method")
    ax.set_title(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(root / f"{key}_bar.png", dpi=160)
    plt.close(fig)


def plot_training_curves(root: Path, all_metrics: dict[str, dict[str, Any]]) -> None:
    plt = import_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    curve_keys = (
        ("train_loss", "Train loss"),
        ("stimulus_top1", "Stimulus top-1"),
        ("class_top1", "Class top-1"),
        ("stimulus_rdm_spearman", "Stimulus RDM Spearman"),
    )
    for ax, (key, title) in zip(axes.flatten(), curve_keys, strict=True):
        for method in METHODS:
            epochs = all_metrics[method].get("epochs", [])
            xs = [int(epoch["epoch"]) for epoch in epochs]
            ys = [float(epoch[key]) for epoch in epochs if key in epoch]
            if xs and ys:
                ax.plot(xs[: len(ys)], ys, marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(root / "training_curves.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot S1 comparison metrics.")
    parser.add_argument("--root", default="outputs/comparison_s1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    all_metrics = {method: load_metrics(root / method / "metrics.json") for method in METHODS}
    rows = write_summary(root, all_metrics)
    plot_bar(root, rows, "stimulus_top1", "Stimulus top-1")
    plot_bar(root, rows, "stimulus_top5", "Stimulus top-5")
    plot_bar(root, rows, "class_top1", "Class top-1")
    plot_bar(root, rows, "stimulus_rdm_spearman", "Stimulus RDM Spearman")
    plot_bar(root, rows, "class_rdm_spearman", "Class RDM Spearman")
    plot_training_curves(root, all_metrics)
    print({"summary": str(root / "summary.csv"), "rows": rows})


if __name__ == "__main__":
    main()
