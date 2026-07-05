"""LLM pathway-prior conditioned perturbation model.

Architecture:
  control expression -> cell encoder -> cell tokens
  fp * log10(dose+1) -> drug encoder -> drug tokens
  14-dim LLM pathway prior -> prior encoder -> prior tokens

  drug-effect representation = concat(drug tokens, prior tokens)
  Q = cell tokens
  K, V = drug-effect representation
  cross-attention -> perturbed hidden -> decoder -> perturbed expression
"""

from __future__ import annotations

import torch
import torch.nn as nn

PROGENY_PATHWAY_NAMES = (
    "Androgen",
    "EGFR",
    "Estrogen",
    "Hypoxia",
    "JAK-STAT",
    "MAPK",
    "NFkB",
    "PI3K",
    "TGFb",
    "TNFa",
    "Trail",
    "VEGF",
    "WNT",
    "p53",
)


class TokenMLPEncoder(nn.Module):
    """Map a flat input vector to multiple context tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_tokens: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.to_tokens = nn.Linear(in_dim, num_tokens * hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        tokens = self.to_tokens(h)
        batch_size = x.shape[0]
        return tokens.view(batch_size, self.num_tokens, self.hidden_dim)


class ConcatMLPFusion(nn.Module):
    """Ablation: fuse cell and drug-effect tokens without cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_cell_tokens: int,
        num_drug_tokens: int,
        num_prior_tokens: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_tokens = num_cell_tokens + num_drug_tokens + num_prior_tokens
        self.net = nn.Sequential(
            nn.Linear(n_tokens * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        drug_effect_tokens: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([query_tokens, drug_effect_tokens], dim=1)
        flat = combined.reshape(combined.shape[0], -1)
        hidden = self.net(flat)
        return hidden.unsqueeze(1)


class CrossAttentionFusion(nn.Module):
    """Cell tokens query drug-effect representation."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        drug_effect_tokens: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(drug_effect_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        h = query_tokens + attn_out
        h = h + self.ff(self.norm_out(h))
        return h


class PerturbationDecoder(nn.Module):
    """Map perturbed hidden tokens to gene expression delta."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for _ in range(max(1, num_layers - 1)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        pooled = hidden_tokens.mean(dim=1)
        return self.net(pooled)


class PrismPerturbModel(nn.Module):
    """Cell queries drug-effect representation to predict perturbed expression."""

    def __init__(
        self,
        gene_size: int,
        output_dim: int,
        drug_dimension: int = 1024,
        n_pathways: int = len(PROGENY_PATHWAY_NAMES),
        hidden_dim: int = 2048,
        num_heads: int = 4,
        num_cell_tokens: int = 1,
        num_drug_tokens: int = 1,
        num_prior_tokens: int = 1,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        predict_delta: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.gene_size = int(gene_size)
        self.output_dim = int(output_dim)
        self.drug_dimension = int(drug_dimension)
        self.n_pathways = int(n_pathways)
        self.hidden_dim = int(hidden_dim)
        self.predict_delta = bool(predict_delta)
        self.use_cross_attention = bool(use_cross_attention)
        self.backbone = "prism"
        self.num_cell_tokens = int(num_cell_tokens)
        self.num_drug_tokens = int(num_drug_tokens)
        self.num_prior_tokens = int(num_prior_tokens)

        self.cell_encoder = TokenMLPEncoder(
            input_dim=gene_size,
            hidden_dim=hidden_dim,
            num_tokens=num_cell_tokens,
            num_layers=encoder_layers,
            dropout=dropout,
        )
        self.drug_encoder = TokenMLPEncoder(
            input_dim=drug_dimension,
            hidden_dim=hidden_dim,
            num_tokens=num_drug_tokens,
            num_layers=encoder_layers,
            dropout=dropout,
        )
        self.prior_encoder = TokenMLPEncoder(
            input_dim=n_pathways,
            hidden_dim=hidden_dim,
            num_tokens=num_prior_tokens,
            num_layers=encoder_layers,
            dropout=dropout,
        )
        if self.use_cross_attention:
            self.cross_attn = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.fusion_mlp = None
        else:
            self.cross_attn = None
            self.fusion_mlp = ConcatMLPFusion(
                hidden_dim=hidden_dim,
                num_cell_tokens=num_cell_tokens,
                num_drug_tokens=num_drug_tokens,
                num_prior_tokens=num_prior_tokens,
                dropout=dropout,
            )
        self.decoder = PerturbationDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def encode_tokens(
        self,
        control_feature: torch.Tensor,
        drug_dose: torch.Tensor,
        pathway_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_tokens = self.cell_encoder(control_feature)
        drug_tokens = self.drug_encoder(drug_dose)
        prior_tokens = self.prior_encoder(pathway_prior)
        return cell_tokens, drug_tokens, prior_tokens

    def build_drug_effect_representation(
        self,
        drug_tokens: torch.Tensor,
        prior_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([drug_tokens, prior_tokens], dim=1)

    def forward(
        self,
        control_feature: torch.Tensor,
        drug_dose: torch.Tensor,
        pathway_prior: torch.Tensor,
        return_hidden: bool = False,
    ):
        cell_tokens, drug_tokens, prior_tokens = self.encode_tokens(
            control_feature, drug_dose, pathway_prior
        )
        drug_effect = self.build_drug_effect_representation(drug_tokens, prior_tokens)
        if self.use_cross_attention:
            perturbed_hidden = self.cross_attn(cell_tokens, drug_effect)
        else:
            perturbed_hidden = self.fusion_mlp(cell_tokens, drug_effect)
        delta = self.decoder(perturbed_hidden)
        if self.predict_delta:
            prediction = control_feature + delta
        else:
            prediction = delta
        if return_hidden:
            return prediction, delta, perturbed_hidden
        return prediction, delta

    def to_checkpoint(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "backbone": "prism",
            "gene_size": self.gene_size,
            "output_dim": self.output_dim,
            "drug_dimension": self.drug_dimension,
            "n_pathways": self.n_pathways,
            "hidden_dim": self.hidden_dim,
            "predict_delta": self.predict_delta,
            "use_cross_attention": self.use_cross_attention,
            "pathway_names": list(PROGENY_PATHWAY_NAMES),
            "architecture": "cell_query_drug_effect",
        }


def load_prism_checkpoint(path: str, device, **model_kwargs) -> PrismPerturbModel:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        meta = payload
        state = payload["state_dict"]
        model = PrismPerturbModel(
            gene_size=meta.get("gene_size", model_kwargs.get("gene_size", 200)),
            output_dim=meta.get("output_dim", model_kwargs.get("output_dim", 200)),
            drug_dimension=meta.get(
                "drug_dimension", model_kwargs.get("drug_dimension", 1024)
            ),
            n_pathways=meta.get(
                "n_pathways", model_kwargs.get("n_pathways", len(PROGENY_PATHWAY_NAMES))
            ),
            hidden_dim=meta.get("hidden_dim", model_kwargs.get("hidden_dim", 2048)),
            predict_delta=meta.get(
                "predict_delta", model_kwargs.get("predict_delta", True)
            ),
            use_cross_attention=meta.get(
                "use_cross_attention", model_kwargs.get("use_cross_attention", True)
            ),
        )
        model.load_state_dict(state, strict=False)
        missing = set(model.state_dict()) - set(state)
        if missing:
            print(
                f"Loaded checkpoint with {len(missing)} newly initialized parameter(s): "
                f"{sorted(missing)[:5]}...",
                flush=True,
            )
        return model.to(device)

    model = PrismPerturbModel(**model_kwargs)
    model.load_state_dict(payload, strict=True)
    return model.to(device)
