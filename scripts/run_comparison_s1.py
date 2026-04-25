from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrgs.data import MRGSProcessedDataset, load_feature_tensor
from mrgs.train import DEFAULT_CONFIG, run_training
from mrgs.utils import deep_update, load_config


METHODS: dict[str, dict[str, float]] = {
    "contrastive_only": {"lambda_ot": 0.0, "lambda_rdm": 0.0},
    "contrastive_ot": {"lambda_ot": 1.0, "lambda_rdm": 0.0},
    "mrgs": {"lambda_ot": 1.0, "lambda_rdm": 10.0},
}

REPORT_KEYS = (
    "retrieval_top1",
    "retrieval_top5",
    "retrieval_mean_rank",
    "retrieval_median_rank",
    "rdm_spearman",
)


def is_fallback_stimulus_ids(image_paths: list[str], subject: str, split: str) -> bool:
    prefix = f"{subject}:{split}:stimulus_"
    return bool(image_paths) and all(path.startswith(prefix) for path in image_paths)


def class_ids_look_like_fallback(class_ids: torch.Tensor) -> bool:
    expected = torch.arange(class_ids.numel(), dtype=class_ids.dtype)
    return class_ids.cpu().equal(expected)


def strict_split_check(processed_path: Path, feature_path: Path, subject: str, split: str) -> dict[str, Any]:
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed split: {processed_path}")
    if not feature_path.exists():
        payload = torch.load(processed_path, map_location="cpu")
        if is_fallback_stimulus_ids(list(payload["image_paths"]), subject, split):
            raise FileNotFoundError(
                f"{processed_path} uses fallback stimulus IDs and {feature_path} is missing. "
                "Provide aligned precomputed features before training."
            )
        raise FileNotFoundError(f"Missing feature file: {feature_path}")

    dataset = MRGSProcessedDataset(processed_path, feature_path=feature_path)
    features = load_feature_tensor(feature_path, expected_image_paths=dataset.payload["image_paths"])
    if features.shape[0] != len(dataset):
        raise ValueError(
            f"Feature length mismatch for {split}: {features.shape[0]} features vs {len(dataset)} processed samples"
        )
    if is_fallback_stimulus_ids(dataset.payload["image_paths"], subject, split):
        print(f"INFO: {split} image_paths are fallback stimulus IDs; using aligned feature file {feature_path}.")
    if class_ids_look_like_fallback(dataset.payload["class_ids"]):
        print(f"WARNING: {split} class_ids look like fallback arange IDs.")
    return {
        "split": split,
        "samples": len(dataset),
        "fmri_dim": dataset.spec.fmri_dim,
        "image_feature_dim": int(features.shape[1]),
        "classes": dataset.spec.num_classes,
        "processed": str(processed_path),
        "features": str(feature_path),
    }


def validate_real_data(processed_dir: Path, feature_dir: Path, subject: str = "S1") -> dict[str, Any]:
    train = strict_split_check(
        processed_dir / f"{subject}_train.pt",
        feature_dir / f"{subject}_train_features.pt",
        subject,
        "train",
    )
    test = strict_split_check(
        processed_dir / f"{subject}_test.pt",
        feature_dir / f"{subject}_test_features.pt",
        subject,
        "test",
    )
    print({"train": train, "test": test})
    return {"train": train, "test": test}


def final_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    final = metrics["epochs"][-1] if metrics.get("epochs") else metrics["initial_eval"]
    return {key: float(final[key]) for key in REPORT_KEYS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S1 loss-ablation comparison on real processed data.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--config", default="configs/minimal_clip.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--output-root", default="outputs/comparison_s1")
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--roi", default="HVC")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    feature_dir = Path(args.feature_dir)
    output_root = Path(args.output_root)
    validation = validate_real_data(processed_dir, feature_dir, subject=args.subject)

    base_config = deep_update(DEFAULT_CONFIG, load_config(args.config))
    if args.seed is not None:
        base_config["seed"] = args.seed

    summaries: dict[str, dict[str, float]] = {}
    for method, loss_overrides in METHODS.items():
        method_output = output_root / method
        if method_output.exists() and args.overwrite:
            shutil.rmtree(method_output)
        config = deep_update(
            base_config,
            {
                "device": args.device,
                "data": {
                    "subject": args.subject,
                    "roi": args.roi,
                    "data_dir": str(processed_dir),
                    "features_dir": str(feature_dir),
                    "synthetic": False,
                },
                "loss": loss_overrides,
                "train": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "output_dir": str(output_root / method),
                },
            },
        )
        print(f"RUN {method}: {loss_overrides}")
        metrics = run_training(config)
        inner_metrics = method_output / args.subject / "metrics.json"
        target_metrics = method_output / "metrics.json"
        if inner_metrics.exists():
            shutil.copy2(inner_metrics, target_metrics)
        else:
            target_metrics.parent.mkdir(parents=True, exist_ok=True)
            target_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summaries[method] = final_metrics(metrics)

    print({"validation": validation, "final": summaries})


if __name__ == "__main__":
    main()
