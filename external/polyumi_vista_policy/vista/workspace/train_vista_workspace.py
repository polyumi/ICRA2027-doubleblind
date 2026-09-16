"""Hydra workspace for Vista multimodal policy training."""

from __future__ import annotations

import copy
import json
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import dill
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from vista.policy.base import BaseVistaPolicy
from vista.scripts.param_report import format_split_report, param_split

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainVistaWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BaseVistaPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: Optional[BaseVistaPolicy] = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        encoder_lr = cfg.optimizer.get("encoder_lr", None)
        if encoder_lr is not None:
            enc_params = list(self.model.encoder_parameters())
            head_params = list(self.model.head_parameters())
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop("_target_")
            base_lr = optimizer_cfg.pop("lr")
            # Pop before splatting into AdamW (encoder_lr is not an AdamW kwarg).
            encoder_lr = optimizer_cfg.pop("encoder_lr") or base_lr
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": head_params, "lr": base_lr},
                    {"params": enc_params, "lr": encoder_lr},
                ],
                **optimizer_cfg,
            )
        else:
            opt_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            opt_cfg.pop("encoder_lr", None)
            self.optimizer = hydra.utils.instantiate(
                OmegaConf.create(opt_cfg), params=self.model.parameters()
            )

        self.global_step = 0
        self.epoch = 0
        if not cfg.training.resume:
            self.exclude_keys = ["optimizer"]

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        accelerator = Accelerator(log_with="wandb")
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, "wb") as f:
                pickle.dump(normalizer, f)
            split = param_split(self.model)
            print(format_split_report(type(self.model).__name__, split))
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            with open(normalizer_path, "rb") as f:
                normalizer = pickle.load(f)
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        max_epochs = cfg.training.get("max_epochs")
        if max_epochs is None:
            max_epochs = cfg.training.get("num_epochs")

        steps_per_epoch = len(train_dataloader) // cfg.training.gradient_accumulate_every
        if max_epochs is None:
            lr_scheduler = get_scheduler(
                "constant_with_warmup",
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=steps_per_epoch * max_epochs,
            )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = (
            accelerator.prepare(
                train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
            )
        )
        if self.ema_model is not None:
            self.ema_model.to(accelerator.device)

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        json_logger = JsonLogger(log_path)
        json_logger.start()

        epoch = 0
        while max_epochs is None or epoch < max_epochs:
            self.model.train()
            train_losses = []
            with tqdm(train_dataloader, desc=f"epoch {epoch}", leave=False) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(accelerator.device))
                    raw_loss = self.model(batch)
                    loss = raw_loss["loss"] if isinstance(raw_loss, dict) else raw_loss
                    loss = loss / cfg.training.gradient_accumulate_every
                    accelerator.backward(loss)
                    if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                        if ema is not None:
                            ema.step(accelerator.unwrap_model(self.model))
                    train_losses.append(loss.item())
                    self.global_step += 1

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = dict_apply(batch, lambda x: x.to(accelerator.device))
                    out = self.model(batch)
                    val_losses.append(out["loss"].item() if isinstance(out, dict) else out.item())
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss

            log_data = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            json_logger.log(log_data)
            accelerator.log(log_data, step=self.global_step)
            self.epoch = epoch

            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                if accelerator.is_main_process:
                    ckpt_path = topk_manager.get_ckpt_path(log_data)
                    if ckpt_path is not None:
                        self.save_checkpoint(path=ckpt_path, use_thread=False)
                    self.save_checkpoint(use_thread=False)
                    epoch_ckpt = Path(self.output_dir).joinpath(
                        "checkpoints", f"epoch={epoch:04d}.ckpt"
                    )
                    self.save_checkpoint(path=epoch_ckpt, use_thread=False)

            epoch += 1

        json_logger.stop()
        accelerator.end_training()
