from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "image_paths",
    "class_ids",
    "class_names",
    "fmri",
    "subject",
    "split",
}


@dataclass(frozen=True)
class DatasetSpec:
    subject: str
    split: str
    fmri_dim: int
    image_feature_dim: int
    size: int
    num_classes: int


def validate_processed_payload(payload: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"Processed payload is missing keys: {sorted(missing)}")
    if not isinstance(payload["image_paths"], list):
        raise TypeError("image_paths must be a list[str]")
    if not torch.is_tensor(payload["class_ids"]) or payload["class_ids"].dtype != torch.long:
        raise TypeError("class_ids must be a torch.LongTensor")
    if not isinstance(payload["class_names"], list):
        raise TypeError("class_names must be a list[str]")
    if not torch.is_tensor(payload["fmri"]) or not payload["fmri"].is_floating_point():
        raise TypeError("fmri must be a floating point tensor")
    n = payload["fmri"].shape[0]
    if len(payload["image_paths"]) != n or payload["class_ids"].shape[0] != n:
        raise ValueError("image_paths, class_ids, and fmri must have matching first dimensions")


def load_feature_tensor(path: str | Path, expected_image_paths: list[str] | None = None) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu")
    if torch.is_tensor(payload):
        return payload.float()
    if isinstance(payload, dict):
        if expected_image_paths is not None and "image_paths" in payload:
            if list(payload["image_paths"]) != expected_image_paths:
                raise ValueError(f"Feature file image_paths do not match processed file: {path}")
        for key in ("features", "image_features", "embeddings"):
            if key in payload and torch.is_tensor(payload[key]):
                return payload[key].float()
    raise ValueError(f"Could not find a feature tensor in {path}")


class MRGSProcessedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        processed_path: str | Path,
        feature_path: str | Path | None = None,
        require_features: bool = True,
    ) -> None:
        self.processed_path = Path(processed_path)
        self.payload = torch.load(self.processed_path, map_location="cpu")
        validate_processed_payload(self.payload)

        embedded = self.payload.get("image_features")
        if embedded is not None:
            if not torch.is_tensor(embedded):
                raise TypeError("image_features in processed payload must be a tensor")
            self.image_features = embedded.float()
        elif feature_path is not None:
            self.image_features = load_feature_tensor(feature_path, expected_image_paths=self.payload["image_paths"])
        elif require_features:
            raise ValueError(
                "Image features are required. Provide feature_path or store image_features in the processed file."
            )
        else:
            self.image_features = None

        if self.image_features is not None and self.image_features.shape[0] != self.payload["fmri"].shape[0]:
            raise ValueError("image_features and fmri must have matching first dimensions")

    @property
    def spec(self) -> DatasetSpec:
        image_feature_dim = 0 if self.image_features is None else int(self.image_features.shape[1])
        return DatasetSpec(
            subject=str(self.payload["subject"]),
            split=str(self.payload["split"]),
            fmri_dim=int(self.payload["fmri"].shape[1]),
            image_feature_dim=image_feature_dim,
            size=len(self),
            num_classes=int(self.payload["class_ids"].unique().numel()),
        )

    def __len__(self) -> int:
        return int(self.payload["fmri"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "fmri": self.payload["fmri"][index].float(),
            "class_id": self.payload["class_ids"][index].long(),
            "image_path": self.payload["image_paths"][index],
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.image_features is not None:
            item["image_features"] = self.image_features[index].float()
        return item


class SyntheticPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        n_samples: int = 512,
        image_feature_dim: int = 768,
        fmri_dim: int = 1024,
        latent_dim: int = 64,
        num_classes: int = 50,
        subject: str = "S1",
        split: str = "train",
        noise_std: float = 0.05,
        seed: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        class_centers = torch.randn(num_classes, latent_dim, generator=generator)
        class_ids = torch.arange(n_samples) % num_classes
        permutation = torch.randperm(n_samples, generator=generator)
        class_ids = class_ids[permutation]
        latent = class_centers[class_ids] + 0.25 * torch.randn(n_samples, latent_dim, generator=generator)
        image_w = torch.randn(latent_dim, image_feature_dim, generator=generator) / latent_dim**0.5
        fmri_w = torch.randn(latent_dim, fmri_dim, generator=generator) / latent_dim**0.5
        self.image_features = latent @ image_w + noise_std * torch.randn(n_samples, image_feature_dim, generator=generator)
        self.fmri = latent @ fmri_w + noise_std * torch.randn(n_samples, fmri_dim, generator=generator)
        self.class_ids = class_ids.long()
        self.image_paths = [f"synthetic://{split}/{i:06d}.jpg" for i in range(n_samples)]
        self.class_names = [f"class_{i:03d}" for i in range(num_classes)]
        self.subject = subject
        self.split = split

    @property
    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            subject=self.subject,
            split=self.split,
            fmri_dim=int(self.fmri.shape[1]),
            image_feature_dim=int(self.image_features.shape[1]),
            size=len(self),
            num_classes=int(self.class_ids.unique().numel()),
        )

    def __len__(self) -> int:
        return int(self.fmri.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image_features": self.image_features[index].float(),
            "fmri": self.fmri[index].float(),
            "class_id": self.class_ids[index].long(),
            "image_path": self.image_paths[index],
            "index": torch.tensor(index, dtype=torch.long),
        }

    def to_processed_payload(self) -> dict[str, Any]:
        return {
            "image_paths": self.image_paths,
            "class_ids": self.class_ids,
            "class_names": self.class_names,
            "fmri": self.fmri.float(),
            "image_features": self.image_features.float(),
            "subject": self.subject,
            "split": self.split,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "fmri": torch.stack([item["fmri"] for item in batch]),
        "class_ids": torch.stack([item["class_id"] for item in batch]),
        "indices": torch.stack([item["index"] for item in batch]),
        "image_paths": [item["image_path"] for item in batch],
    }
    if "image_features" in batch[0]:
        output["image_features"] = torch.stack([item["image_features"] for item in batch])
    return output


def processed_paths(
    data_dir: str | Path,
    subject: str,
    features_dir: str | Path | None = None,
) -> tuple[Path, Path | None, Path, Path | None]:
    data_root = Path(data_dir)
    train_path = data_root / f"{subject}_train.pt"
    test_path = data_root / f"{subject}_test.pt"
    if features_dir is None:
        return train_path, None, test_path, None
    feature_root = Path(features_dir)
    return (
        train_path,
        feature_root / f"{subject}_train_features.pt",
        test_path,
        feature_root / f"{subject}_test_features.pt",
    )
