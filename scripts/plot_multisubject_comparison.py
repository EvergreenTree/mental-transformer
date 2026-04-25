from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = ("contrastive_only", "contrastive_ot", "mrgs")
METRICS = (
    "stimulus_top1",
    "stimulus_top5",
    "stimulus_rdm_spearman",
    "class_rdm_spearman",
)


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Run must be SUBJECT=ROOT, got: {value}")
    subject, root = value.split("=", 1)
    return subject, Path(root)


def load_summary(subject: str, root: Path) -> list[dict[str, str]]:
    path = root / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["subject"] = subject
            row["root"] = str(root)
            rows.append(row)
    return rows


def metric_values(rows: list[dict[str, str]], method: str, metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if row["method"] == method and row.get(metric) not in (None, "")]


def write_aggregate_summary(output_dir: Path, rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    output_rows: list[dict[str, float | str]] = []
    for method in METHODS:
        row: dict[str, float | str] = {"method": method, "subjects": len(metric_values(rows, method, "stimulus_top1"))}
        for metric in METRICS:
            values = metric_values(rows, method, metric)
            row[f"{metric}_mean"] = sum(values) / len(values)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        for baseline in ("stimulus_random_top1", "stimulus_random_top5"):
            values = metric_values(rows, method, baseline)
            row[f"{baseline}_mean"] = sum(values) / len(values)
        output_rows.append(row)

    fieldnames = list(output_rows[0].keys())
    output = output_dir / "multisubject_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def plot_aggregate(output_dir: Path, rows: list[dict[str, str]], summary_rows: list[dict[str, float | str]]) -> None:
    plt = import_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = (
        ("stimulus_top1", "stimulus_random_top1", "Stimulus top-1"),
        ("stimulus_top5", "stimulus_random_top5", "Stimulus top-5"),
        ("stimulus_rdm_spearman", None, "Stimulus RDM Spearman"),
        ("class_rdm_spearman", None, "Class RDM Spearman"),
    )
    for ax, (metric, baseline, title) in zip(axes.flatten(), panels, strict=True):
        means = [float(row[f"{metric}_mean"]) for row in summary_rows]
        mins = [float(row[f"{metric}_min"]) for row in summary_rows]
        maxs = [float(row[f"{metric}_max"]) for row in summary_rows]
        lower = [mean - lo for mean, lo in zip(means, mins, strict=True)]
        upper = [hi - mean for mean, hi in zip(means, maxs, strict=True)]
        ax.bar(METHODS, means, yerr=[lower, upper], capsize=4, alpha=0.75)
        for index, method in enumerate(METHODS):
            values = metric_values(rows, method, metric)
            ax.scatter([index] * len(values), values, color="black", s=18, zorder=3)
        if baseline is not None:
            baseline_values = [float(row[f"{baseline}_mean"]) for row in summary_rows]
            chance = sum(baseline_values) / len(baseline_values)
            ax.axhline(chance, color="black", linestyle="--", linewidth=1, label=f"random {chance:.3g}")
            ax.legend()
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "multisubject_stimulus_summary.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metric-fixed comparison plots across subjects.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Subject/root pair, e.g. S1=outputs/comparison_s1_vgg19_vc_metricfix",
    )
    parser.add_argument("--output-dir", default="outputs/comparison_all_subjects_vgg19_vc_metricfix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for value in args.run:
        subject, root = parse_run(value)
        rows.extend(load_summary(subject, root))

    subject_rows = output_dir / "multisubject_subject_rows.csv"
    with subject_rows.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["subject", "root", *rows[0].keys()]
        fieldnames = list(dict.fromkeys(fieldnames))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = write_aggregate_summary(output_dir, rows)
    plot_aggregate(output_dir, rows, summary_rows)
    print(
        {
            "subject_rows": str(subject_rows),
            "summary": str(output_dir / "multisubject_summary.csv"),
            "plot": str(output_dir / "multisubject_stimulus_summary.png"),
        }
    )


if __name__ == "__main__":
    main()
