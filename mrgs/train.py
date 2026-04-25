from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrgs.data import MRGSProcessedDataset, SyntheticPairDataset, collate_batch, processed_paths
from mrgs.eval import evaluate_model
from mrgs.losses import LossWeights, combined_mrgs_loss
from mrgs.models import MRGSModel
from mrgs.utils import deep_update, load_config, save_checkpoint, save_json, seed_everything, select_device


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "data_dir": "data/processed",
        "features_dir": None,
        "subject": "S1",
        "roi": "HVC",
        "synthetic": False,
        "num_workers": 0,
    },
    "synthetic": {
        "train_samples": 512,
        "test_samples": 128,
        "image_feature_dim": 256,
        "fmri_dim": 384,
        "latent_source_dim": 64,
        "num_classes": 50,
        "noise_std": 0.05,
    },
    "model": {
        "latent_dim": 512,
        "hidden_dim": 1024,
        "projector_layers": 2,
        "encoder_layers": 2,
        "dropout": 0.1,
        "use_attention": False,
        "attention_heads": 8,
    },
    "loss": {
        "temperature": 0.07,
        "tau": 0.07,
        "lambda_ot": 1.0,
        "lambda_rdm": 10.0,
        "sinkhorn_iters": 20,
    },
    "train": {
        "epochs": 20,
        "batch_size": 64,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "eval_batch_size": 128,
        "output_dir": "outputs",
        "save_every": 0,
    },
}


def build_datasets(config: dict[str, Any]) -> tuple[Any, Any]:
    data_cfg = config["data"]
    if data_cfg.get("synthetic", False):
        syn_cfg = config["synthetic"]
        train = SyntheticPairDataset(
            n_samples=syn_cfg["train_samples"],
            image_feature_dim=syn_cfg["image_feature_dim"],
            fmri_dim=syn_cfg["fmri_dim"],
            latent_dim=syn_cfg["latent_source_dim"],
            num_classes=syn_cfg["num_classes"],
            subject=data_cfg["subject"],
            split="train",
            noise_std=syn_cfg["noise_std"],
            seed=config["seed"],
        )
        test = SyntheticPairDataset(
            n_samples=syn_cfg["test_samples"],
            image_feature_dim=syn_cfg["image_feature_dim"],
            fmri_dim=syn_cfg["fmri_dim"],
            latent_dim=syn_cfg["latent_source_dim"],
            num_classes=syn_cfg["num_classes"],
            subject=data_cfg["subject"],
            split="test",
            noise_std=syn_cfg["noise_std"],
            seed=config["seed"] + 1,
        )
        return train, test

    train_path, train_features, test_path, test_features = processed_paths(
        data_cfg["data_dir"],
        data_cfg["subject"],
        data_cfg.get("features_dir"),
    )
    return (
        MRGSProcessedDataset(train_path, feature_path=train_features),
        MRGSProcessedDataset(test_path, feature_path=test_features),
    )


def build_model_from_dataset(config: dict[str, Any], dataset: Any, device: torch.device) -> MRGSModel:
    spec = dataset.spec
    model_cfg = config["model"]
    model = MRGSModel(
        image_feature_dim=spec.image_feature_dim,
        fmri_dim=spec.fmri_dim,
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        projector_layers=model_cfg["projector_layers"],
        encoder_layers=model_cfg["encoder_layers"],
        dropout=model_cfg["dropout"],
        use_attention=model_cfg["use_attention"],
        attention_heads=model_cfg["attention_heads"],
    )
    return model.to(device)


def train_one_epoch(
    model: MRGSModel,
    dataloader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    weights: LossWeights,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        image_features = batch["image_features"].to(device)
        fmri = batch["fmri"].to(device)
        outputs = model(image_features=image_features, fmri=fmri)
        losses = combined_mrgs_loss(outputs["z_img"], outputs["z_fmri"], weights)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

        batch_size = image_features.shape[0]
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(config["seed"])
    device = select_device(config["device"])
    train_dataset, test_dataset = build_datasets(config)
    model = build_model_from_dataset(config, train_dataset, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_batch,
        drop_last=True,
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    weights = LossWeights(**config["loss"])
    metrics: dict[str, Any] = {
        "subject": config["data"]["subject"],
        "roi": config["data"]["roi"],
        "device": str(device),
        "initial_eval": evaluate_model(model, eval_loader, device),
        "epochs": [],
    }

    output_dir = Path(config["train"]["output_dir"]) / config["data"]["subject"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["train"]["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, weights, device)
        eval_metrics = evaluate_model(model, eval_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **eval_metrics,
        }
        metrics["epochs"].append(epoch_metrics)
        print(epoch_metrics)

        save_every = int(config["train"].get("save_every", 0))
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", model, config, epoch_metrics)

    final_metrics = metrics["epochs"][-1] if metrics["epochs"] else metrics["initial_eval"]
    save_checkpoint(output_dir / "last.pt", model, config, final_metrics)
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal MRGS alignment model.")
    parser.add_argument("--config", default="configs/minimal_clip.yaml")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--processed-dir", dest="data_dir", help="Alias for --data-dir.")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--feature-dir", dest="features_dir", help="Alias for --features-dir.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(DEFAULT_CONFIG, load_config(args.config))
    overrides: dict[str, Any] = {"data": {}, "train": {}}
    if args.subject is not None:
        overrides["data"]["subject"] = args.subject
    if args.roi is not None:
        overrides["data"]["roi"] = args.roi
    if args.data_dir is not None:
        overrides["data"]["data_dir"] = args.data_dir
    if args.features_dir is not None:
        overrides["data"]["features_dir"] = args.features_dir
    if args.synthetic:
        overrides["data"]["synthetic"] = True
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train"]["batch_size"] = args.batch_size
    if args.output_dir is not None:
        overrides["train"]["output_dir"] = args.output_dir
    if args.device is not None:
        overrides["device"] = args.device
    return deep_update(config, overrides)


def main() -> None:
    args = parse_args()
    run_training(config_from_args(args))


if __name__ == "__main__":
    main()
