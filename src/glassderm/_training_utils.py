"""Shared training loop for the 3 CNN-based pipelines.

One loop, one loss schema, one history dict — so MultiTaskCNN / HardCBM /
GlassBoxNAM differ purely in their reasoning head and not in the way they are
optimised.  That is the only fair way to compare them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import dump_json


def _make_optimizer(model: nn.Module, cfg):
    name = cfg.train.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer {cfg.train.optimizer!r}")


def _make_scheduler(optimizer, cfg, steps_per_epoch: int):
    name = (cfg.train.scheduler or "none").lower()
    if name in {"none", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.train.epochs * steps_per_epoch)
        )
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.train.lr,
            epochs=cfg.train.epochs,
            steps_per_epoch=steps_per_epoch,
        )
    raise ValueError(f"Unknown scheduler {cfg.train.scheduler!r}")


def _pos_weight(train_loader: DataLoader) -> torch.Tensor:
    df = train_loader.dataset.df
    n_pos = int(df.label.sum())
    n_neg = int((df.label == 0).sum())
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def train_loop(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    *,
    device: str,
    cfg,
    logger,
    concept_sup_weight: float = 1.0,
    label_loss_weight: float = 1.0,
    concept_loss_weight: float = 1.0,
    tag: str = "model",
) -> Dict[str, Any]:
    """Train the pipeline's CNN model and return a per-epoch history."""
    pos_weight = _pos_weight(loaders["train"]).to(device)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    abcd_loss = nn.MSELoss()

    opt = _make_optimizer(model, cfg)
    sched = _make_scheduler(opt, cfg, len(loaders["train"]))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device == "cuda"))

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(cfg.train.epochs):
        model.train()
        running = {"loss": 0.0, "cls": 0.0, "abcd": 0.0, "n": 0}
        pbar = tqdm(
            loaders["train"],
            desc=f"[{tag}] epoch {epoch+1}/{cfg.train.epochs}",
            leave=False,
        )
        for batch in pbar:
            imgs, labels, concepts, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=(cfg.train.amp and device == "cuda")
            ):
                logits, concept_pred = model(imgs)
                loss_c = cls_loss(logits.view(-1), labels.view(-1))
                loss_a = abcd_loss(concept_pred, concepts)
                loss = (
                    label_loss_weight * loss_c
                    + concept_loss_weight * concept_sup_weight * loss_a
                )

            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()

            bs = imgs.size(0)
            running["loss"] += float(loss.item()) * bs
            running["cls"] += float(loss_c.item()) * bs
            running["abcd"] += float(loss_a.item()) * bs
            running["n"] += bs

            pbar.set_postfix(
                loss=running["loss"] / max(running["n"], 1),
                cls=running["cls"] / max(running["n"], 1),
                abcd=running["abcd"] / max(running["n"], 1),
            )

        val_metrics = _validate(model, loaders["val"], device, cls_loss, abcd_loss)
        history["train"].append({k: v / max(running["n"], 1) for k, v in running.items() if k != "n"})
        history["val"].append(val_metrics)
        logger.info(
            "[%s] epoch %d  train_loss=%.4f  val_loss=%.4f  val_auc=%.3f",
            tag,
            epoch + 1,
            history["train"][-1]["loss"],
            val_metrics["loss"],
            val_metrics["auc"],
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if cfg.outputs.reports:
        dump_json(history, Path(cfg.outputs.reports) / f"history_{tag}.json")

    return history


def _validate(
    model: nn.Module, loader: DataLoader, device: str, cls_loss, abcd_loss
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    model.eval()
    total_loss, total_n = 0.0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, concepts, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            concepts = concepts.to(device)
            logits, concept_pred = model(imgs)
            loss = cls_loss(logits.view(-1), labels.view(-1)) + abcd_loss(
                concept_pred, concepts
            )
            total_loss += float(loss.item()) * imgs.size(0)
            total_n += imgs.size(0)
            all_probs.append(torch.sigmoid(logits.view(-1)).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"loss": total_loss / max(total_n, 1), "auc": float(auc)}


def predict_loader(
    model: nn.Module, loader: DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    model.eval()
    all_probs, all_labels, all_concepts, all_ids = [], [], [], []
    with torch.no_grad():
        for imgs, labels, concepts, ids in loader:
            imgs = imgs.to(device)
            logits, concept_pred = model(imgs)
            all_probs.append(torch.sigmoid(logits.view(-1)).cpu().numpy())
            all_labels.append(labels.cpu().numpy().astype(int))
            all_concepts.append(concept_pred.cpu().numpy())
            all_ids.extend(list(ids))
    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_concepts, axis=0),
        all_ids,
    )
