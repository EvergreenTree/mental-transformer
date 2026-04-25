from __future__ import annotations

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from mrgs.data import SyntheticPairDataset, collate_batch
from mrgs.eval import evaluate_model
from mrgs.losses import LossWeights, combined_mrgs_loss
from mrgs.models import MRGSModel
from mrgs.train import DEFAULT_CONFIG, run_training
from mrgs.utils import deep_update


def test_synthetic_forward_loss_and_metrics() -> None:
    dataset = SyntheticPairDataset(
        n_samples=64,
        image_feature_dim=32,
        fmri_dim=40,
        latent_dim=16,
        num_classes=8,
        seed=123,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    model = MRGSModel(
        image_feature_dim=dataset.spec.image_feature_dim,
        fmri_dim=dataset.spec.fmri_dim,
        latent_dim=24,
        hidden_dim=48,
        dropout=0.0,
    )
    batch = next(iter(loader))
    outputs = model(batch["image_features"], batch["fmri"])
    losses = combined_mrgs_loss(outputs["z_img"], outputs["z_fmri"], LossWeights(lambda_rdm=1.0))
    assert torch.isfinite(losses["loss"])
    metrics = evaluate_model(model, loader, torch.device("cpu"))
    assert {"retrieval_top1", "retrieval_top5", "retrieval_mean_rank", "retrieval_median_rank", "rdm_spearman"}.issubset(
        metrics
    )


def test_synthetic_training_smoke(tmp_path) -> None:
    config = deep_update(
        DEFAULT_CONFIG,
        {
            "seed": 7,
            "device": "cpu",
            "data": {"synthetic": True, "subject": "S1", "num_workers": 0},
            "synthetic": {
                "train_samples": 128,
                "test_samples": 64,
                "image_feature_dim": 48,
                "fmri_dim": 56,
                "latent_source_dim": 16,
                "num_classes": 8,
                "noise_std": 0.02,
            },
            "model": {
                "latent_dim": 32,
                "hidden_dim": 64,
                "projector_layers": 2,
                "encoder_layers": 2,
                "dropout": 0.0,
                "use_attention": False,
                "attention_heads": 4,
            },
            "loss": {"lambda_rdm": 1.0, "sinkhorn_iters": 5},
            "train": {
                "epochs": 2,
                "batch_size": 32,
                "eval_batch_size": 64,
                "output_dir": str(tmp_path),
                "save_every": 0,
            },
        },
    )
    metrics = run_training(config)
    epochs = metrics["epochs"]
    assert len(epochs) == 2
    assert torch.isfinite(torch.tensor(epochs[-1]["train_loss"]))
    assert epochs[-1]["retrieval_top5"] >= 5 / config["synthetic"]["test_samples"]
    assert (tmp_path / "S1" / "last.pt").exists()


def test_feature_extractor_saves_requested_schema(tmp_path, monkeypatch) -> None:
    from scripts import extract_image_features

    image_paths = []
    for index in range(2):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (8, 8), color=(index * 50, 10, 20)).save(path)
        image_paths.append(path)
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("\n".join(str(path) for path in image_paths), encoding="utf-8")
    output = tmp_path / "features.pt"

    class TinyEncoder(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.flatten(start_dim=1)[:, :5]

    def fake_mobilenet(device: torch.device):
        def transform(image: Image.Image) -> torch.Tensor:
            return torch.full((3, 8, 8), float(image.size[0]))

        return TinyEncoder().to(device), transform, "fake_mobilenet"

    monkeypatch.setattr(extract_image_features, "build_mobilenet", fake_mobilenet)
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract_image_features.py",
            "--image-paths",
            str(paths_file),
            "--output",
            str(output),
            "--backend",
            "mobilenet_v3_small",
            "--batch-size",
            "1",
            "--device",
            "cpu",
        ],
    )

    extract_image_features.main()
    payload = torch.load(output, map_location="cpu")
    assert payload["image_paths"] == [str(path) for path in image_paths]
    assert payload["features"].shape == (2, 5)
    assert payload["features"].dtype == torch.float32
    assert payload["backend"] == "mobilenet_v3_small"
    assert payload["model_name"] == "mobilenet_v3_small"
