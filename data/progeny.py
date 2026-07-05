"""PROGENy pathway activity scoring.

The implementation intentionally uses a local CSV file instead of downloading
resources at inference time. Expected columns are:

- pathway: pathway name, e.g. MAPK or p53
- target or gene: target gene symbol
- weight: PROGENy coefficient

Additional columns, such as p_value, are ignored unless ``top_n`` filtering is
requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


PATHWAY_COLUMN_CANDIDATES = ("pathway", "source")
GENE_COLUMN_CANDIDATES = ("target", "gene", "genesymbol")
WEIGHT_COLUMN_CANDIDATES = ("weight", "mor", "coefficient", "coef")
PVALUE_COLUMN_CANDIDATES = ("p_value", "pvalue", "p.value")


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str:
    normalized = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not find any of columns {list(candidates)} in {list(columns)}"
    )


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


@dataclass(frozen=True)
class PathwayCoverage:
    pathway: str
    covered_genes: int
    total_genes: int
    coverage: float


@dataclass(frozen=True)
class ProgenyGeneScore:
    gene: str
    dataset_gene: str
    n_pathways: int
    weight_sum: float
    best_p_value: float | None


@dataclass(frozen=True)
class ProgenyGeneSelection:
    genes: list[str]
    scores: list[ProgenyGeneScore]
    coverage: list[PathwayCoverage]

    def to_dict(self):
        return {
            "genes": self.genes,
            "scores": [score.__dict__ for score in self.scores],
            "coverage": [item.__dict__ for item in self.coverage],
        }


class ProgenyModel:
    """Linear PROGENy scorer aligned to an AnnData gene list."""

    def __init__(
        self,
        weights: pd.DataFrame,
        pathway_col: str = "pathway",
        gene_col: str = "gene",
        weight_col: str = "weight",
        min_genes: int = 1,
    ):
        self.weights = weights.copy()
        self.pathway_col = pathway_col
        self.gene_col = gene_col
        self.weight_col = weight_col
        self.min_genes = min_genes

        self.weights[self.gene_col] = self.weights[self.gene_col].astype(str)
        self.pathways = sorted(self.weights[self.pathway_col].astype(str).unique())
        self._matrix_cache = {}

    @classmethod
    def from_csv(
        cls,
        path: str,
        top_n: Optional[int] = None,
        min_genes: int = 1,
    ) -> "ProgenyModel":
        weights = pd.read_csv(path)
        pathway_col = _find_column(weights.columns, PATHWAY_COLUMN_CANDIDATES)
        gene_col = _find_column(weights.columns, GENE_COLUMN_CANDIDATES)
        weight_col = _find_column(weights.columns, WEIGHT_COLUMN_CANDIDATES)

        keep_cols = [pathway_col, gene_col, weight_col]
        pvalue_col = None
        for candidate in PVALUE_COLUMN_CANDIDATES:
            matches = [col for col in weights.columns if col.lower() == candidate]
            if matches:
                pvalue_col = matches[0]
                keep_cols.append(pvalue_col)
                break

        weights = weights[keep_cols].dropna(subset=[pathway_col, gene_col, weight_col])
        weights = weights.rename(
            columns={pathway_col: "pathway", gene_col: "gene", weight_col: "weight"}
        )
        weights["pathway"] = weights["pathway"].astype(str)
        weights["gene"] = weights["gene"].astype(str).str.upper()
        weights["weight"] = weights["weight"].astype(float)

        if top_n is not None:
            if pvalue_col and pvalue_col in weights.columns:
                weights = (
                    weights.sort_values(["pathway", pvalue_col])
                    .groupby("pathway", sort=False)
                    .head(top_n)
                )
            else:
                weights = (
                    weights.assign(abs_weight=weights["weight"].abs())
                    .sort_values(["pathway", "abs_weight"], ascending=[True, False])
                    .groupby("pathway", sort=False)
                    .head(top_n)
                    .drop(columns=["abs_weight"])
                )

        return cls(
            weights=weights,
            pathway_col="pathway",
            gene_col="gene",
            weight_col="weight",
            min_genes=min_genes,
        )

    def build_weight_matrix(self, gene_names: Sequence[str]) -> np.ndarray:
        """Return a genes x pathways weight matrix aligned to ``gene_names``."""
        cache_key = tuple(str(gene).upper() for gene in gene_names)
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]

        gene_to_idx = {gene: idx for idx, gene in enumerate(cache_key)}
        pathway_to_idx = {name: idx for idx, name in enumerate(self.pathways)}
        matrix = np.zeros((len(gene_names), len(self.pathways)), dtype=np.float32)

        gene_set = set(gene_to_idx)
        relevant = self.weights[
            self.weights[self.gene_col].astype(str).str.upper().isin(gene_set)
        ]
        for row in relevant.itertuples(index=False):
            gene = getattr(row, self.gene_col).upper()
            if gene not in gene_to_idx:
                continue
            pathway = getattr(row, self.pathway_col)
            matrix[gene_to_idx[gene], pathway_to_idx[pathway]] += float(
                getattr(row, self.weight_col)
            )

        self._matrix_cache[cache_key] = matrix
        return matrix

    def coverage(self, gene_names: Sequence[str]) -> list[PathwayCoverage]:
        genes = {str(gene).upper() for gene in gene_names}
        result = []
        for pathway in self.pathways:
            pathway_weights = self.weights[self.weights[self.pathway_col] == pathway]
            total = pathway_weights[self.gene_col].nunique()
            covered = pathway_weights[
                pathway_weights[self.gene_col].str.upper().isin(genes)
            ][self.gene_col].nunique()
            result.append(
                PathwayCoverage(
                    pathway=pathway,
                    covered_genes=int(covered),
                    total_genes=int(total),
                    coverage=float(covered / total) if total else 0.0,
                )
            )
        return result

    def active_pathways(self, gene_names: Sequence[str]) -> list[str]:
        return [
            item.pathway
            for item in self.coverage(gene_names)
            if item.covered_genes >= self.min_genes
        ]

    def transform(
        self,
        expression,
        gene_names: Sequence[str],
        center: bool = False,
    ) -> np.ndarray:
        """Map expression rows to pathway activities."""
        x = _to_dense(expression).astype(np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != len(gene_names):
            raise ValueError(
                f"Expression has {x.shape[1]} genes, but {len(gene_names)} names were given"
            )

        if center:
            x = x - x.mean(axis=0, keepdims=True)

        weight_matrix = self.build_weight_matrix(gene_names)
        activities = x @ weight_matrix

        coverage = self.coverage(gene_names)
        inactive = np.array(
            [item.covered_genes < self.min_genes for item in coverage], dtype=bool
        )
        activities[:, inactive] = np.nan
        return activities

    def transform_adata(self, adata, center: bool = False) -> np.ndarray:
        return self.transform(adata.X, adata.var_names, center=center)

    def delta(
        self,
        treatment_expression,
        control_expression,
        gene_names: Sequence[str],
        center: bool = False,
    ) -> np.ndarray:
        treatment = self.transform(treatment_expression, gene_names, center=center)
        control = self.transform(control_expression, gene_names, center=center)
        if treatment.shape != control.shape:
            raise ValueError(
                f"Treatment activity shape {treatment.shape} != control {control.shape}"
            )
        return treatment - control

    def delta_adata(self, treatment_adata, control_adata, center: bool = False):
        if list(treatment_adata.var_names) != list(control_adata.var_names):
            raise ValueError("Treatment and control AnnData gene names must match")
        return self.delta(
            treatment_adata.X,
            control_adata.X,
            treatment_adata.var_names,
            center=center,
        )


def load_progeny(path: str, top_n: Optional[int] = None, min_genes: int = 1):
    return ProgenyModel.from_csv(path, top_n=top_n, min_genes=min_genes)


def _build_gene_lookup(gene_names: Sequence[str]) -> dict[str, str]:
    lookup = {}
    for gene in gene_names:
        key = str(gene).upper()
        if key not in lookup:
            lookup[key] = str(gene)
    return lookup


def rank_progeny_genes(
    progeny: ProgenyModel,
    candidate_gene_names: Sequence[str],
) -> list[ProgenyGeneScore]:
    """Rank dataset genes that appear in the PROGENy weight table."""
    lookup = _build_gene_lookup(candidate_gene_names)
    weights = progeny.weights.copy()
    weights["gene_upper"] = weights[progeny.gene_col].astype(str).str.upper()
    weights = weights[weights["gene_upper"].isin(lookup)]
    if weights.empty:
        return []

    grouped = weights.groupby("gene_upper", sort=False)
    scores = []
    for gene_upper, frame in grouped:
        p_values = None
        if "p_value" in frame.columns:
            p_values = frame["p_value"].astype(float)
        scores.append(
            ProgenyGeneScore(
                gene=gene_upper,
                dataset_gene=lookup[gene_upper],
                n_pathways=int(frame[progeny.pathway_col].nunique()),
                weight_sum=float(frame[progeny.weight_col].abs().sum()),
                best_p_value=float(p_values.min()) if p_values is not None else None,
            )
        )

    scores.sort(
        key=lambda item: (
            -item.n_pathways,
            -item.weight_sum,
            item.best_p_value if item.best_p_value is not None else float("inf"),
            item.dataset_gene,
        )
    )
    return scores


def select_progeny_genes(
    progeny_path: str,
    candidate_gene_names: Sequence[str],
    n_genes: int = 200,
    top_n: Optional[int] = None,
    min_genes: int = 1,
) -> ProgenyGeneSelection:
    """Select the top ``n_genes`` PROGENy genes present in ``candidate_gene_names``."""
    progeny = load_progeny(progeny_path, top_n=top_n, min_genes=min_genes)
    ranked = rank_progeny_genes(progeny, candidate_gene_names)
    if len(ranked) < n_genes:
        raise ValueError(
            f"Only {len(ranked)} PROGENy genes overlap the dataset, "
            f"but {n_genes} were requested."
        )

    selected_scores = ranked[:n_genes]
    selected_genes = [score.dataset_gene for score in selected_scores]
    coverage = progeny.coverage(selected_genes)
    return ProgenyGeneSelection(
        genes=selected_genes,
        scores=selected_scores,
        coverage=coverage,
    )
