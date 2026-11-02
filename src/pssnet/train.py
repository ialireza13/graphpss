"""Train and evaluate one configuration.

    python -m pssnet.train --config configs/pssnet_gatv2.yaml [--seed 1] [--set train.max_epochs=2]

Early stopping and checkpoint selection use the validation split; the test split is
evaluated once, with the best checkpoint, at the end.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import CLASS_NAMES, NUM_CLASSES, GraphStore, HopStore, load_or_make_split, make_loader
from .models import build_model


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    loss_sum, n = 0.0, 0
    per_protein = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(batch)
        loss_sum += nn.functional.cross_entropy(logits.float(), batch.y, reduction="sum").item()
        n += batch.y.numel()
        pred = logits.argmax(1)
        idx = batch.y * NUM_CLASSES + pred
        conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
        hits = torch.bincount(batch.batch, weights=(pred == batch.y).float(), minlength=batch.num_graphs)
        sizes = torch.bincount(batch.batch, minlength=batch.num_graphs).clamp(min=1)
        per_protein.append((hits / sizes).cpu())
    return metrics_from_confusion(conf.cpu().numpy(), loss_sum / n,
                                  per_protein_acc=float(torch.cat(per_protein).mean()))


def metrics_from_confusion(conf, loss=None, per_protein_acc=None):
    tp = np.diag(conf).astype(float)
    precision = tp / np.maximum(conf.sum(0), 1)
    recall = tp / np.maximum(conf.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    out = {
        "acc": float(tp.sum() / conf.sum()),
        "macro_f1": float(f1.mean()),
        "f1": {c: float(v) for c, v in zip(CLASS_NAMES, f1)},
        "confusion": conf.tolist(),
    }
    if loss is not None:
        out["loss"] = float(loss)
    if per_protein_acc is not None:
        out["per_protein_acc"] = per_protein_acc
    return out


def run(cfg, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "seed": seed}, f)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc, dc = cfg["train"], cfg["data"]
    amp = tc.get("amp", True) and device.type == "cuda"

    store = GraphStore.load(dc["store"])
    split = load_or_make_split(dc["split"], len(store), dc.get("split_seed", 0))
    if tc.get("subset"):  # for smoke tests
        split = {k: v[: tc["subset"]] for k, v in split.items()}
    if tc.get("train_fraction", 1.0) < 1.0:  # for the hyperparameter sweep; val/test stay whole
        rng = np.random.default_rng(1234)
        split["train"] = rng.permutation(split["train"])[: int(tc["train_fraction"] * len(split["train"]))].tolist()
    hops = cfg["model"].get("hops", 1)
    hop_store = HopStore(dc["hops"]) if hops > 1 and dc.get("hops") and os.path.exists(dc["hops"]) else None
    workers = tc.get("workers", 8)
    bs = tc["batch_size"]
    loader = lambda ids, b, shuffle, s=0: make_loader(store, ids, b, shuffle, hops, workers, s,  # noqa: E731
                                                     hop_store=hop_store)
    train_loader = loader(split["train"], bs, True, seed)
    val_loader = loader(split["val"], 2 * bs, False)
    test_loader = loader(split["test"], 2 * bs, False)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["max_epochs"], eta_min=tc.get("eta_min", 1e-4))
    print(f"[{out_dir}] params={n_params/1e6:.2f}M device={device} amp={amp}", flush=True)

    history, best, best_state, bad_epochs = [], float("inf"), None, 0
    t_start = time.time()
    for epoch in range(tc["max_epochs"]):
        model.train()
        t0 = time.time()
        loss_sum, correct, n = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(batch)
            loss = nn.functional.cross_entropy(logits.float(), batch.y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.get("clip", 1.0))
            opt.step()
            loss_sum += loss.item() * batch.y.numel()
            correct += (logits.argmax(1) == batch.y).sum().item()
            n += batch.y.numel()
        sched.step()
        val = evaluate(model, val_loader, device, amp)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "train_acc": correct / n,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
               "lr": opt.param_groups[0]["lr"], "epoch_sec": time.time() - t0}
        improved = val["loss"] < best
        if improved:
            best, bad_epochs = val["loss"], 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_state, "config": cfg, "epoch": epoch}, os.path.join(out_dir, "best.pt"))
        else:
            bad_epochs += 1
        history.append(row)
        print(f"ep {epoch:3d} train {row['train_loss']:.4f}/{row['train_acc']:.4f} "
              f"val {row['val_loss']:.4f}/{row['val_acc']:.4f} f1 {row['val_macro_f1']:.4f} "
              f"{'*' if improved else ' '} {row['epoch_sec']:.0f}s", flush=True)
        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        if bad_epochs >= tc["patience"]:
            print("early stopping", flush=True)
            break
        if tc.get("max_hours") and (time.time() - t_start) / 3600 > tc["max_hours"]:
            print("time budget reached", flush=True)
            break

    model.load_state_dict(best_state)
    results = {
        "name": cfg.get("name"), "seed": seed, "params": n_params,
        "best_epoch": int(np.argmin([h["val_loss"] for h in history])),
        "epochs_run": len(history), "train_hours": (time.time() - t_start) / 3600,
        "val": evaluate(model, val_loader, device, amp),
        "test": evaluate(model, test_loader, device, amp),
    }
    if tc.get("eval_train", True):
        results["train"] = evaluate(model, loader(split["train"], 2 * bs, False), device, amp)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"TEST acc={results['test']['acc']:.4f} macro_f1={results['test']['macro_f1']:.4f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. train.lr=5e-4")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = args.out or os.path.join(cfg.get("runs_dir", "runs"), f"{cfg['name']}_s{args.seed}")
    run(cfg, args.seed, out)


if __name__ == "__main__":
    main()
