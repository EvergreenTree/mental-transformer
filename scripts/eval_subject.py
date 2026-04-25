from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrgs.data import MRGSProcessedDataset, collate_batch
from mrgs.eval import evaluate_model
from mrgs.models import MRGSModel
from mrgs.utils import load_checkpoint, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained minimal MRGS checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed", required=True, help="Processed S*_test.pt file.")
    parser.add_argument("--features", default=None, help="Optional precomputed feature tensor.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    dataset = MRGSProcessedDataset(args.processed, feature_path=args.features)
    spec = dataset.spec
    model_cfg = checkpoint["config"]["model"]
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    print({"checkpoint": str(Path(args.checkpoint)), **evaluate_model(model, loader, device)})


if __name__ == "__main__":
    main()
