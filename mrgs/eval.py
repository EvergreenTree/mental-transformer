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
    if z_img.shape[0] < 2 or z_fmri.shape[0] < 2:
        return float("nan")
    r_img = 1.0 - z_img @ z_img.T
    r_fmri = 1.0 - z_fmri @ z_fmri.T
    return spearman_correlation(upper_triangular_values(r_img), upper_triangular_values(r_fmri))


def _first_match_ranks(order: torch.Tensor, target_equal: torch.Tensor) -> torch.Tensor:
    ranked_hits = torch.gather(target_equal, 1, order)
    hit_positions = ranked_hits.float().argmax(dim=1) + 1
    if not bool(ranked_hits.any(dim=1).all()):
        raise ValueError("Every query must have at least one valid retrieval target.")
    return hit_positions.long()


def _rank_metrics(ranks: torch.Tensor, n_targets: int, prefix: str) -> dict[str, float]:
    ranks_f = ranks.float()
    return {
        f"{prefix}_top1": float((ranks <= 1).float().mean().item()),
        f"{prefix}_top5": float((ranks <= min(5, n_targets)).float().mean().item()),
        f"{prefix}_mean_rank": float(ranks_f.mean().item()),
        f"{prefix}_median_rank": float(ranks_f.median().item()),
    }


def _group_retrieval_metrics(
    sim: torch.Tensor,
    query_group_ids: list[Any],
    target_group_ids: list[Any],
    prefix: str,
) -> dict[str, float]:
    if len(query_group_ids) != sim.shape[0] or len(target_group_ids) != sim.shape[1]:
        raise ValueError(f"{prefix} group id lengths must match similarity matrix")
    groups = list(dict.fromkeys(target_group_ids))
    group_scores = []
    for group in groups:
        target_indices = [index for index, value in enumerate(target_group_ids) if value == group]
        group_scores.append(sim[:, target_indices].max(dim=1).values)
    grouped_sim = torch.stack(group_scores, dim=1)
    order = torch.argsort(grouped_sim, dim=1, descending=True)
    group_index = {group: index for index, group in enumerate(groups)}
    target_indices = torch.tensor([group_index[value] for value in query_group_ids], device=sim.device).unsqueeze(1)
    group_positions = torch.arange(len(groups), device=sim.device).unsqueeze(0)
    ranks = _first_match_ranks(order, group_positions == target_indices)
    metrics = _rank_metrics(ranks, len(groups), prefix)
    metrics[f"{prefix}_random_top1"] = 1.0 / max(len(groups), 1)
    metrics[f"{prefix}_random_top5"] = min(5, len(groups)) / max(len(groups), 1)
    return metrics


def brain_to_image_retrieval(
    z_img: torch.Tensor,
    z_fmri: torch.Tensor,
    stimulus_ids: list[str] | None = None,
    class_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    sim = z_fmri @ z_img.T
    order = torch.argsort(sim, dim=1, descending=True)
    query_indices = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1)
    target_indices = torch.arange(sim.shape[1], device=sim.device).unsqueeze(0)
    row_ranks = _first_match_ranks(order, target_indices == query_indices)
    metrics = _rank_metrics(row_ranks, sim.shape[1], "row")
    metrics.update(
        {
            "top1": metrics["row_top1"],
            "top5": metrics["row_top5"],
            "mean_rank": metrics["row_mean_rank"],
            "median_rank": metrics["row_median_rank"],
        }
    )

    if stimulus_ids is not None:
        if len(stimulus_ids) != sim.shape[0]:
            raise ValueError("stimulus_ids length must match embeddings")
        metrics.update(_group_retrieval_metrics(sim, stimulus_ids, stimulus_ids, "stimulus"))

    if class_ids is not None:
        if class_ids.shape[0] != sim.shape[0]:
            raise ValueError("class_ids length must match embeddings")
        class_values = [int(value) for value in class_ids.tolist()]
        metrics.update(_group_retrieval_metrics(sim, class_values, class_values, "class"))
    return metrics


def average_by_group(z: torch.Tensor, group_ids: list[Any]) -> torch.Tensor:
    if len(group_ids) != z.shape[0]:
        raise ValueError("group_ids length must match embeddings")
    groups = list(dict.fromkeys(group_ids))
    return torch.stack([z[[index for index, value in enumerate(group_ids) if value == group]].mean(dim=0) for group in groups])


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
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    model.eval()
    img_parts: list[torch.Tensor] = []
    fmri_parts: list[torch.Tensor] = []
    stimulus_ids: list[str] = []
    class_parts: list[torch.Tensor] = []
    for batch in dataloader:
        image_features = batch["image_features"].to(device)
        fmri = batch["fmri"].to(device)
        outputs = model(image_features=image_features, fmri=fmri)
        img_parts.append(outputs["z_img"].detach().cpu())
        fmri_parts.append(outputs["z_fmri"].detach().cpu())
        stimulus_ids.extend(str(path) for path in batch["image_paths"])
        class_parts.append(batch["class_ids"].detach().cpu().long())
    return torch.cat(img_parts, dim=0), torch.cat(fmri_parts, dim=0), stimulus_ids, torch.cat(class_parts, dim=0)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    z_img, z_fmri, stimulus_ids, class_ids = encode_dataset(model, dataloader, device)
    retrieval = brain_to_image_retrieval(z_img, z_fmri, stimulus_ids=stimulus_ids, class_ids=class_ids)
    class_group_ids = [int(value) for value in class_ids.tolist()]
    return {
        **{f"retrieval_{key}": value for key, value in retrieval.items()},
        "row_top1": retrieval["row_top1"],
        "row_top5": retrieval["row_top5"],
        "stimulus_top1": retrieval.get("stimulus_top1", float("nan")),
        "stimulus_top5": retrieval.get("stimulus_top5", float("nan")),
        "stimulus_random_top1": retrieval.get("stimulus_random_top1", float("nan")),
        "stimulus_random_top5": retrieval.get("stimulus_random_top5", float("nan")),
        "class_top1": retrieval.get("class_top1", float("nan")),
        "class_top5": retrieval.get("class_top5", float("nan")),
        "class_random_top1": retrieval.get("class_random_top1", float("nan")),
        "class_random_top5": retrieval.get("class_random_top5", float("nan")),
        "row_rdm_spearman": rdm_correlation(z_img, z_fmri),
        "stimulus_rdm_spearman": rdm_correlation(
            average_by_group(z_img, stimulus_ids),
            average_by_group(z_fmri, stimulus_ids),
        ),
        "class_rdm_spearman": rdm_correlation(
            average_by_group(z_img, class_group_ids),
            average_by_group(z_fmri, class_group_ids),
        ),
        "rdm_spearman": rdm_correlation(z_img, z_fmri),
    }
