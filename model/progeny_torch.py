"""Differentiable PROGENy scorer for Prism training."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class TorchProgeny:
    """Differentiable PROGENy scorer aligned to a fixed gene list."""

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        pathway_active: torch.Tensor,
        pathway_names: list[str],
    ):
        self.weight_matrix = weight_matrix
        self.pathway_active = pathway_active
        self.pathway_names = pathway_names

    @classmethod
    def from_progeny(cls, progeny, gene_names: Sequence[str], device):
        matrix = progeny.build_weight_matrix(gene_names).astype(np.float32)
        coverage = progeny.coverage(gene_names)
        active = np.array(
            [item.covered_genes >= progeny.min_genes for item in coverage],
            dtype=bool,
        )
        return cls(
            weight_matrix=torch.tensor(matrix, dtype=torch.float32, device=device),
            pathway_active=torch.tensor(active, dtype=torch.bool, device=device),
            pathway_names=list(progeny.pathways),
        )

    def activity(self, expression: torch.Tensor, center: bool = False) -> torch.Tensor:
        x = expression
        if center:
            x = x - x.mean(dim=1, keepdim=True)
        activities = x @ self.weight_matrix
        if self.pathway_active.any():
            inactive = ~self.pathway_active
            activities = activities.masked_fill(inactive.view(1, -1), float("nan"))
        return activities

    def delta(
        self,
        treatment: torch.Tensor,
        control: torch.Tensor,
        center: bool = False,
    ) -> torch.Tensor:
        return self.activity(treatment, center=center) - self.activity(
            control, center=center
        )
