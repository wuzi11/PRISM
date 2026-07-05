"""Training loop for PrismPerturbModel."""

from __future__ import annotations

import copy
import os

import torch
from torch.optim import AdamW

from prism.training import dist_util, logger
from prism.training.fp16_util import MixedPrecisionTrainer
from prism.training.nn import update_ema
from prism.training.checkpoint_util import (
    find_ema_checkpoint,
    find_resume_checkpoint,
    parse_resume_step_from_filename,
)
from .prism_losses import compute_prism_losses
from .prism_model import PrismPerturbModel


class PrismTrainLoop:
    def __init__(
        self,
        *,
        model: PrismPerturbModel,
        data,
        batch_size: int,
        microbatch: int,
        lr: float,
        ema_rate,
        log_interval: int,
        save_interval: int,
        resume_checkpoint: str,
        use_fp16: bool = False,
        fp16_scale_growth: float = 1e-3,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        use_drug_structure: bool = True,
        torch_progeny=None,
        expr_loss_type: str = "mse",
        pathway_loss_weight: float = 1.0,
        pathway_margin: float = 0.0,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.use_drug_structure = use_drug_structure
        self.torch_progeny = torch_progeny
        self.expr_loss_type = expr_loss_type
        self.pathway_loss_weight = float(pathway_loss_weight)
        self.pathway_margin = float(pathway_margin)

        self.step = 0
        self.resume_step = 0
        self.loss_list = []

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self._load_and_sync_parameters()
        if self.resume_step:
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        load_path = None
        if self.resume_checkpoint:
            if os.path.isdir(self.resume_checkpoint):
                candidate = os.path.join(self.resume_checkpoint, "model.pt")
                if os.path.isfile(candidate):
                    load_path = candidate
            elif os.path.isfile(self.resume_checkpoint):
                load_path = self.resume_checkpoint
        if load_path:
            self.resume_step = parse_resume_step_from_filename(load_path)
            logger.log(f"loading Prism model from checkpoint: {load_path}...")
            payload = torch.load(load_path, map_location=dist_util.dev())
            if isinstance(payload, dict) and "state_dict" in payload:
                self.model.load_state_dict(payload["state_dict"])
            else:
                self.model.load_state_dict(payload)

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint and os.path.isdir(main_checkpoint):
            candidate = os.path.join(main_checkpoint, "model.pt")
            main_checkpoint = candidate if os.path.isfile(candidate) else None
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            state_dict = torch.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        return ema_params

    def run_loop(self):
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            batch = next(iter(self.data))
            self.run_step(batch)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        if self.step % self.log_interval == 0:
            logger.logkv("step", self.step + self.resume_step)

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        device = dist_util.dev()
        total_loss = torch.zeros((), device=device)
        n_seen = 0

        for i in range(0, batch["feature"].shape[0], self.microbatch):
            micro = batch["feature"][i : i + self.microbatch].to(device)
            if self.use_drug_structure:
                drug = batch["drug_dose"][i : i + self.microbatch].to(device)
                control = batch["control_feature"][i : i + self.microbatch].to(device)
            else:
                raise ValueError("Prism model requires --use_drug_structure")

            if "pathway_prior" in batch:
                pathway_prior = batch["pathway_prior"][i : i + self.microbatch].to(device)
            else:
                pathway_prior = torch.zeros(
                    micro.shape[0],
                    self.model.n_pathways,
                    device=device,
                    dtype=micro.dtype,
                )

            losses = compute_prism_losses(
                model=self.model,
                target_expression=micro,
                control_feature=control,
                drug_dose=drug,
                pathway_prior=pathway_prior,
                torch_progeny=self.torch_progeny,
                expr_loss_type=self.expr_loss_type,
                pathway_loss_weight=self.pathway_loss_weight,
                pathway_margin=self.pathway_margin,
            )
            loss = losses["loss"]
            self.mp_trainer.backward(loss)

            total_loss = total_loss + loss.detach() * micro.shape[0]
            n_seen += micro.shape[0]
            logger.logkv_mean("expr_loss", float(losses["expr_loss"].detach().cpu()))
            logger.logkv_mean("pathway_loss", float(losses["pathway_loss"].detach().cpu()))
            logger.logkv_mean("loss", float(loss.detach().cpu()))

        self.loss_list.append(total_loss / max(n_seen, 1))

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * max(0.0, 1.0 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        os.makedirs(self.resume_checkpoint, exist_ok=True)

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if rate:
                filepath = os.path.join(self.resume_checkpoint, f"model_{rate}.pt")
            else:
                filepath = os.path.join(self.resume_checkpoint, "model.pt")
            if rate == 0.9999 or not rate:
                payload = self.model.to_checkpoint()
                payload["state_dict"] = state_dict
                torch.save(payload, filepath)
            else:
                torch.save(state_dict, filepath)
            logger.log(f"saved Prism checkpoint to {filepath}")

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
