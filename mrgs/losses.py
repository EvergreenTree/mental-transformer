from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    temperature: float = 0.07
    tau: float = 0.07
    lambda_ot: float = 1.0
    lambda_rdm: float = 10.0
    sinkhorn_iters: int = 20


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return log_alpha.exp()


def contrastive_pairing_loss(
    z_img: torch.Tensor,
    z_fmri: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = z_img @ z_fmri.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def sinkhorn_ot_loss(
    z_img: torch.Tensor,
    z_fmri: torch.Tensor,
    tau: float = 0.07,
    n_iters: int = 20,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    sim = z_img @ z_fmri.T
    transport = sinkhorn(sim / tau, n_iters=n_iters)
    loss = -torch.log(torch.diagonal(transport).clamp_min(eps)).mean()
    return loss, transport


def rdm_alignment_loss(z_img: torch.Tensor, z_fmri: torch.Tensor) -> torch.Tensor:
    r_img = 1.0 - z_img @ z_img.T
    r_fmri = 1.0 - z_fmri @ z_fmri.T
    return F.mse_loss(r_img, r_fmri)


def combined_mrgs_loss(
    z_img: torch.Tensor,
    z_fmri: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    loss_clip = contrastive_pairing_loss(z_img, z_fmri, temperature=weights.temperature)
    loss_ot, transport = sinkhorn_ot_loss(
        z_img,
        z_fmri,
        tau=weights.tau,
        n_iters=weights.sinkhorn_iters,
    )
    loss_rdm = rdm_alignment_loss(z_img, z_fmri)
    loss = loss_clip + weights.lambda_ot * loss_ot + weights.lambda_rdm * loss_rdm
    return {
        "loss": loss,
        "loss_clip": loss_clip.detach(),
        "loss_ot": loss_ot.detach(),
        "loss_rdm": loss_rdm.detach(),
        "transport_diag_mean": torch.diagonal(transport).mean().detach(),
    }
