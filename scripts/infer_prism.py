#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prism inference: direct forward pass on test cells."""

import argparse
import os

import numpy as np
import scanpy as sc
import torch

from prism.model.prism_model import load_prism_checkpoint
from prism.model.prism_utils import (
    load_pathway_prior_store,
    tensorize_pathway_priors,
)
from prism.data.scrna_datasets import Drug_dose_encoder


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Prism inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pathway_prior_path", required=True)
    parser.add_argument("--test_adata_path", required=True)
    parser.add_argument("--control_adata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="prism")
    parser.add_argument("--gene_size", type=int, default=200)
    parser.add_argument("--output_dim", type=int, default=200)
    parser.add_argument("--use_drug_structure", type=str2bool, default=True)
    parser.add_argument("--comb_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_cells", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def predict_batched(model, control_x, drug_x, pathway_prior, batch_size, device):
    n_cells = control_x.shape[0]
    outputs = []
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        control = torch.tensor(control_x[start:end], dtype=torch.float32, device=device)
        drug = torch.tensor(drug_x[start:end], dtype=torch.float32, device=device)
        prior = torch.tensor(pathway_prior[start:end], dtype=torch.float32, device=device)
        pred, _ = model(
            control_feature=control,
            drug_dose=drug,
            pathway_prior=prior,
        )
        outputs.append(pred.cpu())
    return torch.cat(outputs, dim=0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    test_adata = sc.read_h5ad(args.test_adata_path)
    control_adata = sc.read_h5ad(args.control_adata_path)
    if args.max_cells and args.max_cells > 0:
        test_adata = test_adata[: args.max_cells].copy()
        control_adata = control_adata[: args.max_cells].copy()
    if list(test_adata.var_names) != list(control_adata.var_names):
        raise ValueError("Test and control gene names must match")

    prior_store = load_pathway_prior_store(args.pathway_prior_path)
    groups = test_adata.obs["Group"].astype(str).tolist()
    pathway_prior = tensorize_pathway_priors(
        groups,
        prior_store["priors"],
        pathway_names=prior_store["pathway_names"],
    ).numpy()

    model = load_prism_checkpoint(
        args.model_path,
        device=device,
        gene_size=args.gene_size,
        output_dim=args.output_dim,
    )
    model.eval()

    control_x = to_dense(control_adata.X).astype(np.float32)
    if args.use_drug_structure:
        drug_x = Drug_dose_encoder(
            test_adata.obs["SMILES"].to_list(),
            test_adata.obs["dose"].to_list(),
            comb_num=args.comb_num,
        ).astype(np.float32)
    else:
        raise ValueError("Prism inference requires --use_drug_structure")

    print(
        f"Prism inference: {test_adata.n_obs} cells (LLM prior + cross-attention)",
        flush=True,
    )
    predictions = predict_batched(
        model=model,
        control_x=control_x,
        drug_x=drug_x,
        pathway_prior=pathway_prior,
        batch_size=args.batch_size,
        device=device,
    )

    tag = f"_{args.tag}" if args.tag else ""
    pred_path = os.path.join(args.output_dir, f"perturbed_expression{tag}.npy")
    np.save(pred_path, predictions.numpy())
    print(
        f"Saved perturbed expression ({predictions.shape}) to {pred_path}\n"
        f"  (control + delta, NOT delta-only)",
        flush=True,
    )


if __name__ == "__main__":
    main()
