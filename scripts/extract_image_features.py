from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrgs.data import MRGSProcessedDataset
from mrgs.utils import select_device


class ImagePathDataset(Dataset[tuple[str, torch.Tensor]]):
    def __init__(self, image_paths: list[str], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return path, tensor


def collate_images(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], torch.Tensor]:
    return [item[0] for item in batch], torch.stack([item[1] for item in batch])


def build_open_clip(
    model_name: str,
    pretrained: str,
    device: torch.device,
) -> tuple[nn.Module, Callable[[Image.Image], torch.Tensor], str]:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("Install open_clip_torch to use --backend open_clip.") from exc

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess, f"{model_name}:{pretrained}"


def build_mobilenet(device: torch.device) -> tuple[nn.Module, Callable[[Image.Image], torch.Tensor], str]:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()
    return model, weights.transforms(), "mobilenet_v3_small:ImageNet1K_V1"


def encode_batch(model: nn.Module, images: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "open_clip":
        return model.encode_image(images)
    if backend == "mobilenet_v3_small":
        return model(images)
    raise ValueError(f"Unsupported backend: {backend}")


@torch.no_grad()
def extract_features(
    image_paths: list[str],
    backend: str,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: torch.device,
    fp16: bool = False,
) -> torch.Tensor:
    if backend == "open_clip":
        model, transform, resolved_name = build_open_clip(model_name, pretrained, device)
    elif backend == "mobilenet_v3_small":
        model, transform, resolved_name = build_mobilenet(device)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    loader = DataLoader(
        ImagePathDataset(image_paths, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_images,
    )
    parts: list[torch.Tensor] = []
    for _, images in tqdm(loader, desc=f"extract:{resolved_name}"):
        batch = images.to(device)
        features = encode_batch(model, batch, backend=backend)
        features = features.detach().cpu().float()
        if fp16:
            features = features.half()
        parts.append(features)
    return torch.cat(parts, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen image features for processed DIR splits.")
    parser.add_argument("--processed", default=None, help="Processed S*_train.pt or S*_test.pt file.")
    parser.add_argument("--image-paths", default=None, help="Text file with one image path per line.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["open_clip", "mobilenet_v3_small"], default="open_clip")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true", help="Save CPU features as float16.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.processed is None) == (args.image_paths is None):
        raise ValueError("Provide exactly one of --processed or --image-paths.")

    if args.processed is not None:
        dataset = MRGSProcessedDataset(args.processed, require_features=False)
        image_paths = list(dataset.payload["image_paths"])
    else:
        image_paths = [
            line.strip()
            for line in Path(args.image_paths).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image paths are missing; first missing path: {missing[0]}. "
            "Provide local DIR/ImageNet subset paths or use cached feature tensors."
        )
    device = select_device(args.device)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 8 if device.type == "mps" else 64

    features = extract_features(
        image_paths,
        backend=args.backend,
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=batch_size,
        device=device,
        fp16=args.fp16,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name if args.backend == "open_clip" else "mobilenet_v3_small"
    torch.save(
        {
            "image_paths": image_paths,
            "features": features,
            "backend": args.backend,
            "model_name": model_name,
            "pretrained": args.pretrained if args.backend == "open_clip" else "ImageNet1K_V1",
        },
        output,
    )
    print(
        {
            "output": str(output),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "backend": args.backend,
            "model_name": model_name,
            "batch_size": batch_size,
            "device": str(device),
        }
    )


if __name__ == "__main__":
    main()
