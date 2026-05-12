"""Per-modality sparse mixture-of-experts feed-forward block."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MoEStats:
    """Routing diagnostics for a Phase 4 MoE block."""

    aux_loss: Tensor
    active_tokens: int


class PerModMoEBlock(nn.Module):
    """Shared experts with one router per modality id.

    The block receives already-normalized tokens and returns a feed-forward
    update. Absent tokens are excluded from routing and load-balancing stats.
    """

    def __init__(
        self,
        *,
        d_model: int = 256,
        n_modalities: int,
        n_experts: int = 8,
        top_k: int = 2,
        expert_ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if top_k < 1 or top_k > n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.n_experts = n_experts
        self.top_k = top_k
        hidden = d_model * expert_ff_mult
        self.routers = nn.ModuleList(
            nn.Linear(d_model, n_experts, bias=False) for _ in range(n_modalities)
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        )
        self.last_stats: MoEStats | None = None

    def forward(
        self,
        tokens: Tensor,
        *,
        modality_ids: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Route ``tokens`` and return ``(update, load_balance_loss)``."""
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (B, T, D); got {tuple(tokens.shape)}")
        batch_size, n_tokens, d_model = tokens.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")
        if modality_ids.shape != (n_tokens,):
            raise ValueError(
                f"modality_ids must have shape ({n_tokens},); got {tuple(modality_ids.shape)}"
            )
        if key_padding_mask is None:
            active_mask = torch.ones(batch_size, n_tokens, dtype=torch.bool, device=tokens.device)
        else:
            if key_padding_mask.shape != (batch_size, n_tokens):
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"({batch_size}, {n_tokens}); got {tuple(key_padding_mask.shape)}"
                )
            active_mask = ~key_padding_mask.bool()

        update = torch.zeros_like(tokens)
        aux_terms: list[Tensor] = []
        active_count = 0
        for token_idx in range(n_tokens):
            token_active = active_mask[:, token_idx]
            if not bool(token_active.any()):
                continue
            modality_id = int(modality_ids[token_idx].item())
            if modality_id < 0 or modality_id >= self.n_modalities:
                raise ValueError(f"modality id {modality_id} outside [0, {self.n_modalities})")
            active_tokens = tokens[token_active, token_idx]
            routed, aux = self._route_one_modality(active_tokens, modality_id)
            update[token_active, token_idx] = routed
            aux_terms.append(aux)
            active_count += int(active_tokens.shape[0])

        aux_loss = torch.stack(aux_terms).mean() if aux_terms else tokens.new_zeros(())
        self.last_stats = MoEStats(aux_loss=aux_loss.detach(), active_tokens=active_count)
        return update, aux_loss

    def _route_one_modality(self, tokens: Tensor, modality_id: int) -> tuple[Tensor, Tensor]:
        logits = self.routers[modality_id](tokens)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        routed = torch.zeros_like(tokens)
        for rank in range(self.top_k):
            expert_ids = top_idx[:, rank]
            expert_weights = top_probs[:, rank]
            for expert_id in expert_ids.unique(sorted=True).tolist():
                selected = expert_ids == int(expert_id)
                expert_out = self.experts[int(expert_id)](tokens[selected])
                routed[selected] += expert_out * expert_weights[selected].unsqueeze(-1)

        hard = F.one_hot(top_idx, num_classes=self.n_experts).float().sum(dim=1)
        hard = hard / float(self.top_k)
        density = hard.mean(dim=0)
        density_proxy = probs.mean(dim=0)
        aux_loss = float(self.n_experts) * torch.sum(density * density_proxy)
        return routed, aux_loss
