from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def _rankdata(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return rankdata(values)
    except Exception:
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(values) + 1)
        return ranks


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    ar = _rankdata(a)
    br = _rankdata(b)
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def upper_triangular_values(matrix: torch.Tensor) -> np.ndarray:
    n = matrix.shape[0]
    indices = torch.triu_indices(n, n, offset=1, device=matrix.device)
    return matrix[indices[0], indices[1]].detach().cpu().numpy()


def rdm_correlation(z_img: torch.Tensor, z_fmri: torch.Tensor) -> float:
    r_img = 1.0 - z_img @ z_img.T
    r_fmri = 1.0 - z_fmri @ z_fmri.T
    return spearman_correlation(upper_triangular_values(r_img), upper_triangular_values(r_fmri))


def brain_to_image_retrieval(z_img: torch.Tensor, z_fmri: torch.Tensor) -> dict[str, float]:
    sim = z_fmri @ z_img.T
    order = torch.argsort(sim, dim=1, descending=True)
    targets = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1)
    ranks = (order == targets).nonzero()[:, 1] + 1
    ranks_f = ranks.float()
    return {
        "top1": float((ranks <= 1).float().mean().item()),
        "top5": float((ranks <= min(5, sim.shape[1])).float().mean().item()),
        "mean_rank": float(ranks_f.mean().item()),
        "median_rank": float(ranks_f.median().item()),
    }


def image_to_image_neighbors(
    z_img: torch.Tensor,
    query_index: int,
    k: int = 5,
) -> list[tuple[int, float]]:
    sim = z_img[query_index] @ z_img.T
    sim[query_index] = -torch.inf
    values, indices = torch.topk(sim, k=min(k, z_img.shape[0] - 1))
    return [(int(idx), float(score)) for idx, score in zip(indices.cpu(), values.cpu(), strict=True)]


@torch.no_grad()
def encode_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    img_parts: list[torch.Tensor] = []
    fmri_parts: list[torch.Tensor] = []
    for batch in dataloader:
        image_features = batch["image_features"].to(device)
        fmri = batch["fmri"].to(device)
        outputs = model(image_features=image_features, fmri=fmri)
        img_parts.append(outputs["z_img"].detach().cpu())
        fmri_parts.append(outputs["z_fmri"].detach().cpu())
    return torch.cat(img_parts, dim=0), torch.cat(fmri_parts, dim=0)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    z_img, z_fmri = encode_dataset(model, dataloader, device)
    retrieval = brain_to_image_retrieval(z_img, z_fmri)
    return {
        **{f"retrieval_{key}": value for key, value in retrieval.items()},
        "rdm_spearman": rdm_correlation(z_img, z_fmri),
    }
