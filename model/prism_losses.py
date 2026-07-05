"""Loss functions for LLM pathway-prior conditioned perturbation model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prism.model.progeny_torch import TorchProgeny


def pathway_prior_consistency_loss(
    pathway_delta: torch.Tensor,
    llm_prior: torch.Tensor,
    margin: float = 0.0,
    pathway_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage PROGENy delta sign to match LLM prior in {-1, 0, 1}.

    prior=1  -> encourage pathway_delta > margin
    prior=-1 -> encourage pathway_delta < -margin
    prior=0  -> no constraint

    Pathways without PROGENy gene coverage (NaN delta) are skipped.
    """
    active = llm_prior != 0
    if pathway_active is not None:
        active = active & pathway_active.view(1, -1)
    active = active & torch.isfinite(pathway_delta)
    if not bool(active.any()):
        return pathway_delta.new_zeros(())
    agreement = pathway_delta * llm_prior
    return F.relu(float(margin) - agreement[active]).mean()


def compute_prism_losses(
    model,
    target_expression: torch.Tensor,
    control_feature: torch.Tensor,
    drug_dose: torch.Tensor,
    pathway_prior: torch.Tensor,
    torch_progeny: TorchProgeny | None = None,
    expr_loss_type: str = "mse",
    pathway_loss_weight: float = 1.0,
    pathway_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    prediction, delta = model(
        control_feature=control_feature,
        drug_dose=drug_dose,
        pathway_prior=pathway_prior,
    )
    if expr_loss_type == "huber":
        expr_loss = F.smooth_l1_loss(prediction, target_expression)
    else:
        expr_loss = F.mse_loss(prediction, target_expression)

    pathway_loss = prediction.new_zeros(())
    if torch_progeny is not None and pathway_loss_weight > 0:
        pathway_delta = torch_progeny.delta(prediction, control_feature)
        pathway_loss = pathway_prior_consistency_loss(
            pathway_delta,
            pathway_prior,
            margin=pathway_margin,
            pathway_active=torch_progeny.pathway_active,
        )

    total_loss = expr_loss + float(pathway_loss_weight) * pathway_loss
    return {
        "loss": total_loss,
        "expr_loss": expr_loss,
        "pathway_loss": pathway_loss,
        "delta": delta,
        "prediction": prediction,
    }
