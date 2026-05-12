"""Phase 4 multitask, reconstruction, and distillation losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

CDR_VALUES = (0.0, 0.5, 1.0, 2.0, 3.0)


def cdr_to_class(cdr: Tensor) -> Tensor:
    """Map CDR scalar values ``0,0.5,1,2,3`` to classes ``0..4``."""
    labels = torch.full_like(cdr, fill_value=-1, dtype=torch.long)
    for idx, value in enumerate(CDR_VALUES):
        labels = torch.where(torch.isclose(cdr.float(), cdr.new_tensor(value)), idx, labels)
    return labels


def compute_phase4_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    teacher_outputs: Mapping[str, Any] | None = None,
    distill_temperature: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute weighted Phase 4 objective and detached diagnostics."""
    cls = _tensor(outputs["cls"])
    device = cls.device
    zero = cls.new_zeros(())
    parts: dict[str, Tensor] = {}

    diag_loss = _diag_loss(_tensor(outputs["diag_logits"]), batch, device, zero)
    surv_loss = _survival_loss(_tensor(outputs["surv_logits"]), batch, device, zero)
    mmse_loss = _masked_regression_loss(
        _tensor(outputs["mmse_pred"]),
        batch,
        target_key="mmse",
        mask_key="has_mmse",
        device=device,
        zero=zero,
    )
    cdr_loss = _cdr_loss(_tensor(outputs["cdr_logits"]), batch, device, zero)
    recon_loss = _reconstruction_loss(outputs, device, zero)
    dist_logit_loss, dist_emb_loss = _distillation_loss(
        outputs,
        teacher_outputs,
        temperature=distill_temperature,
        zero=zero,
    )
    load_balance_loss = outputs.get("aux_loss", zero)
    load_balance = (
        _tensor(load_balance_loss).to(device=device) if load_balance_loss is not None else zero
    )

    raw_losses = {
        "diag": diag_loss,
        "surv": surv_loss,
        "mmse": mmse_loss,
        "cdr": cdr_loss,
        "recon": recon_loss,
        "distill_logits": dist_logit_loss,
        "distill_embedding": dist_emb_loss,
        "load_balance": load_balance,
    }
    total = zero
    for name, value in raw_losses.items():
        parts[name] = value.detach()
        total = total + _weight(weights, name) * value
    parts["total"] = total.detach()
    return total, parts


def _diag_loss(
    logits: Tensor, batch: Mapping[str, Any], device: torch.device, zero: Tensor
) -> Tensor:
    labels = _batch_tensor(batch, "diag_label", device).long()
    mask = labels >= 0
    if not bool(mask.any()):
        return zero
    confidence = _batch_tensor(batch, "label_confidence_weight", device).float()
    per_item = F.cross_entropy(logits, labels.clamp_min(0), reduction="none")
    weights = confidence[mask].clamp_min(0.0)
    return (per_item[mask] * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _survival_loss(
    logits: Tensor,
    batch: Mapping[str, Any],
    device: torch.device,
    zero: Tensor,
) -> Tensor:
    labels = _batch_tensor(batch, "surv_bin", device).long()
    mask = _batch_tensor(batch, "has_survival", device).bool()
    if not bool(mask.any()):
        return zero
    return F.cross_entropy(logits[mask], labels[mask])


def _masked_regression_loss(
    prediction: Tensor,
    batch: Mapping[str, Any],
    *,
    target_key: str,
    mask_key: str,
    device: torch.device,
    zero: Tensor,
) -> Tensor:
    mask = _batch_tensor(batch, mask_key, device).bool()
    if not bool(mask.any()):
        return zero
    target = _batch_tensor(batch, target_key, device).float()
    return F.smooth_l1_loss(prediction[mask].float(), target[mask], reduction="mean")


def _cdr_loss(
    logits: Tensor, batch: Mapping[str, Any], device: torch.device, zero: Tensor
) -> Tensor:
    cdr = _batch_tensor(batch, "cdr", device).float()
    labels = cdr_to_class(cdr)
    mask = _batch_tensor(batch, "has_cdr", device).bool() & (labels >= 0)
    if not bool(mask.any()):
        return zero
    return F.cross_entropy(logits[mask], labels[mask])


def _reconstruction_loss(outputs: Mapping[str, Any], device: torch.device, zero: Tensor) -> Tensor:
    recon = outputs.get("recon", {})
    original = outputs.get("original_tokens", {})
    presence_orig = outputs.get("presence_orig")
    presence_now = outputs.get("presence_now")
    modality_names = outputs.get("modality_names", ())
    if not isinstance(recon, Mapping) or not isinstance(original, Mapping):
        return zero
    if not isinstance(presence_orig, Tensor) or not isinstance(presence_now, Tensor):
        return zero
    losses: list[Tensor] = []
    orig_mask = presence_orig.to(device=device, dtype=torch.bool)
    now_mask = presence_now.to(device=device, dtype=torch.bool)
    for idx, name in enumerate(tuple(str(item) for item in modality_names)):
        if name not in recon or name not in original:
            continue
        mask = orig_mask[:, idx] & ~now_mask[:, idx]
        if not bool(mask.any()):
            continue
        pred = _tensor(recon[name]).to(device=device)
        target = _tensor(original[name]).to(device=device).detach()
        losses.append(F.mse_loss(pred[mask].float(), target[mask].float(), reduction="mean"))
    return torch.stack(losses).mean() if losses else zero


def _distillation_loss(
    outputs: Mapping[str, Any],
    teacher_outputs: Mapping[str, Any] | None,
    *,
    temperature: float,
    zero: Tensor,
) -> tuple[Tensor, Tensor]:
    if teacher_outputs is None:
        return zero, zero
    temp = max(float(temperature), 1.0e-6)
    logit_losses: list[Tensor] = []
    for key in ("diag_logits", "surv_logits"):
        if key not in outputs or key not in teacher_outputs:
            continue
        student = _tensor(outputs[key]).float()
        teacher = _tensor(teacher_outputs[key]).to(device=student.device).float().detach()
        logit_losses.append(
            F.kl_div(
                F.log_softmax(student / temp, dim=-1),
                F.softmax(teacher / temp, dim=-1),
                reduction="batchmean",
            )
            * temp
            * temp
        )
    logit_loss = torch.stack(logit_losses).mean() if logit_losses else zero
    if "cls" not in teacher_outputs:
        return logit_loss, zero
    cls = _tensor(outputs["cls"]).float()
    teacher_cls = _tensor(teacher_outputs["cls"]).to(device=cls.device).float().detach()
    return logit_loss, F.mse_loss(cls, teacher_cls, reduction="mean")


def _weight(weights: Mapping[str, float], name: str) -> float:
    return float(weights.get(name, 0.0))


def _batch_tensor(batch: Mapping[str, Any], key: str, device: torch.device) -> Tensor:
    return _tensor(batch[key]).to(device=device)


def _tensor(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else torch.as_tensor(value)
