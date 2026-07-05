#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train PrismPerturbModel: LLM pathway prior + cross-attention perturbation."""

import argparse
import os
from datetime import datetime
from pathlib import Path

from prism.training import dist_util, logger
from prism.model.prism_model import PrismPerturbModel
from prism.model.prism_train_util import PrismTrainLoop
from prism.data.progeny import load_progeny
from prism.model.progeny_torch import TorchProgeny
from prism.data.scrna_datasets import prepared_data


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logger_path", default="logger_files/prism")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--control_data_path", required=True)
    parser.add_argument("--pathway_prior_path", required=True)
    parser.add_argument(
        "--progeny_path",
        default=str(Path(__file__).resolve().parents[1] / "resources" / "progeny_human.csv"),
    )
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--gene_size", type=int, default=200)
    parser.add_argument("--output_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_cell_tokens", type=int, default=1)
    parser.add_argument("--num_drug_tokens", type=int, default=1)
    parser.add_argument("--num_prior_tokens", type=int, default=1)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--predict_delta", type=str2bool, default=True)
    parser.add_argument("--use_drug_structure", type=str2bool, default=True)
    parser.add_argument("--comb_num", type=int, default=1)
    parser.add_argument("--drug_dimension", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--microbatch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_anneal_steps", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--ema_rate", default="0.9999")
    parser.add_argument("--expr_loss_type", choices=["mse", "huber"], default="mse")
    parser.add_argument("--pathway_loss_weight", type=float, default=1.0)
    parser.add_argument("--pathway_margin", type=float, default=0.0)
    parser.add_argument("--use_cross_attention", type=str2bool, default=True)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    args = parser.parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.logger_path)
    device = dist_util.dev()

    logger.log("creating PrismPerturbModel...")
    model = PrismPerturbModel(
        gene_size=args.gene_size,
        output_dim=args.output_dim,
        drug_dimension=args.drug_dimension,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_cell_tokens=args.num_cell_tokens,
        num_drug_tokens=args.num_drug_tokens,
        num_prior_tokens=args.num_prior_tokens,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        predict_delta=args.predict_delta,
        use_cross_attention=args.use_cross_attention,
    ).to(device)

    progeny = load_progeny(args.progeny_path)
    import scanpy as sc

    train_adata = sc.read_h5ad(args.data_path)
    torch_progeny = TorchProgeny.from_progeny(
        progeny, train_adata.var_names, device=device
    )

    data = prepared_data(
        data_dir=args.data_path,
        control_data_dir=args.control_data_path,
        batch_size=args.batch_size,
        use_drug_structure=args.use_drug_structure,
        comb_num=args.comb_num,
        pathway_prior_path=args.pathway_prior_path,
    )

    out_dir = args.resume_checkpoint or str(
        Path(__file__).resolve().parents[1] / "checkpoints" / f"prism_{Path(args.data_path).stem}"
    )
    os.makedirs(out_dir, exist_ok=True)
    start = datetime.now()
    logger.log(f"Prism training started at {start}")

    PrismTrainLoop(
        model=model,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=out_dir,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_drug_structure=args.use_drug_structure,
        torch_progeny=torch_progeny,
        expr_loss_type=args.expr_loss_type,
        pathway_loss_weight=args.pathway_loss_weight,
        pathway_margin=args.pathway_margin,
    ).run_loop()


if __name__ == "__main__":
    main()
