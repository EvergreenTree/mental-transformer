from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrgs.data import MRGSProcessedDataset
from mrgs.train import DEFAULT_CONFIG, run_training
from mrgs.utils import deep_update, load_config


def run_step(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def has_real_image_paths(processed_path: Path) -> bool:
    payload = torch.load(processed_path, map_location="cpu")
    return all(Path(path).exists() for path in payload["image_paths"])


def validate_features(processed_path: Path, feature_path: Path) -> int:
    dataset = MRGSProcessedDataset(processed_path, feature_path=feature_path)
    return dataset.spec.image_feature_dim


def ensure_features(
    processed_path: Path,
    feature_path: Path,
    backend: str,
    device: str,
    batch_size: int | None,
    fp16: bool,
) -> int:
    if feature_path.exists():
        return validate_features(processed_path, feature_path)
    if not has_real_image_paths(processed_path):
        raise FileNotFoundError(
            f"{feature_path} does not exist and {processed_path} contains stable stimulus IDs rather than local image "
            "paths. Provide precomputed features aligned to the processed file, or update image_paths to local files."
        )
    command = [
        sys.executable,
        "scripts/extract_image_features.py",
        "--processed",
        str(processed_path),
        "--output",
        str(feature_path),
        "--backend",
        backend,
        "--device",
        device,
    ]
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if fp16:
        command.append("--fp16")
    run_step(command)
    return validate_features(processed_path, feature_path)


def processed_summary(path: Path) -> dict[str, Any]:
    dataset = MRGSProcessedDataset(path, require_features=False)
    return {
        "path": str(path),
        "samples": len(dataset),
        "fmri_dim": dataset.spec.fmri_dim,
        "classes": dataset.spec.num_classes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the narrow S1 real-data MRGS path.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--raw", default="data/raw/DIR")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--config", default="configs/minimal_clip.yaml")
    parser.add_argument("--roi", default="HVC")
    parser.add_argument("--backend", default="open_clip", choices=["open_clip", "mobilenet_v3_small"])
    parser.add_argument("--batch-size", type=int, default=None, help="Image feature extraction batch size.")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/real_s1")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--fp16-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw)
    processed_dir = Path(args.processed_dir)
    feature_dir = Path(args.feature_dir)

    if not args.skip_download:
        run_step([sys.executable, "scripts/download_dir_metadata.py", "--download", "--out", str(raw_dir)])

    run_step(
        [
            sys.executable,
            "scripts/prepare_dir.py",
            "--raw",
            str(raw_dir),
            "--out",
            str(processed_dir),
            "--roi",
            args.roi,
            "--subjects",
            "S1",
        ]
    )

    train_processed = processed_dir / "S1_train.pt"
    test_processed = processed_dir / "S1_test.pt"
    train_features = feature_dir / "S1_train_features.pt"
    test_features = feature_dir / "S1_test_features.pt"
    feature_dir.mkdir(parents=True, exist_ok=True)

    train_feature_dim = ensure_features(
        train_processed,
        train_features,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        fp16=args.fp16_features,
    )
    test_feature_dim = ensure_features(
        test_processed,
        test_features,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        fp16=args.fp16_features,
    )

    config = deep_update(DEFAULT_CONFIG, load_config(args.config))
    config = deep_update(
        config,
        {
            "device": args.device,
            "data": {
                "subject": "S1",
                "roi": args.roi,
                "data_dir": str(processed_dir),
                "features_dir": str(feature_dir),
                "synthetic": False,
            },
            "train": {
                "epochs": args.epochs,
                "batch_size": args.train_batch_size,
                "output_dir": args.output_dir,
            },
        },
    )
    metrics = run_training(config)
    final_metrics = metrics["epochs"][-1] if metrics["epochs"] else metrics["initial_eval"]
    print(
        {
            "train": processed_summary(train_processed),
            "test": processed_summary(test_processed),
            "train_feature_dim": train_feature_dim,
            "test_feature_dim": test_feature_dim,
            "metrics": final_metrics,
        }
    )


if __name__ == "__main__":
    main()
