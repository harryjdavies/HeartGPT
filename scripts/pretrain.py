# scripts/pretrain.py

import sys
from pathlib import Path

# ensure that models input properly, without having to make the package installable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import os
import time
import yaml
import logging
import torch
import argparse
from torch.utils.data import DataLoader

from core.model import HeartGPTModel  # adjust if your package path differs
from data.random_sequences import RandomSequenceDataset
from data.ecg_dataset import ECGDataset

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "pretrain.log"))
        ],
    )
    return logging.getLogger("pretrain")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--override-model-path", type=str, default="", help="(Optional) real .mat file for ECG data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output"]["dir"]
    log = setup_logger(out_dir)

    # unpack model hyperparams
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    data_cfg = cfg["data"]
    synth_cfg = cfg["random_sequences"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset selection
    if cfg["data"].get("use_real", False) or args.override_model_path:
        mat_path = args.override_model_path or cfg["data"]["mat_path"]
        train_dataset = ECGDataset(
            mat_path,
            block_size=model_cfg["block_size"],
            split="train",
            train_frac=cfg["data"]["train_frac"],
        )
        val_dataset = ECGDataset(
            mat_path,
            block_size=model_cfg["block_size"],
            split="val",
            train_frac=cfg["data"]["train_frac"],
        )
        log.info("Using real ECG data for pretraining; user must supply their own .mat with 'ecg_store'.")
    else:
        rs_cfg = cfg["random_sequences"]
        train_dataset = RandomSequenceDataset(
            block_size=model_cfg["block_size"],
            vocab_size=model_cfg["vocab_size"],
            base_sequence_count=rs_cfg["base_sequence_count"],
            base_sequence_length=rs_cfg["base_sequence_length"],
            seed=rs_cfg["seed"],
        )
        val_dataset = RandomSequenceDataset(
            block_size=model_cfg["block_size"],
            vocab_size=model_cfg["vocab_size"],
            base_sequence_count=rs_cfg["base_sequence_count"],
            base_sequence_length=rs_cfg["base_sequence_length"],
            seed=rs_cfg["seed"] + 1,
        )
        log.warning("No real ECG/PPG data provided. Falling back to random sequences for bootstrapping.")


    train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg["batch_size"], pin_memory=True)

    # model
    model = HeartGPTModel(
        vocab_size=model_cfg["vocab_size"],
        n_embd=model_cfg["n_embd"],
        n_head=model_cfg["n_head"],
        n_layer=model_cfg["n_layer"],
        block_size=model_cfg["block_size"],
        dropout=model_cfg["dropout"],
    )
    model = model.to(device)

    try:
        learning_rate = float(opt_cfg.get("learning_rate", 3e-4))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid optimizer config: {opt_cfg}") from e

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    def estimate_loss(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                losses.append(loss.item())
                if i + 1 >= training_cfg["eval_iters"]:
                    break
        model.train()
        return sum(losses) / len(losses)

    log.info(f"Starting pretraining. Config: {cfg}")
    log.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    for iteration in range(1, training_cfg["max_iters"] + 1):
        x_batch, y_batch = next(iter(train_loader))
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if iteration % training_cfg["eval_interval"] == 0:
            train_loss = estimate_loss(train_loader)
            val_loss = estimate_loss(val_loader)
            log.info(f"[{iteration}] train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if iteration % training_cfg["save_interval"] == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
                "iteration": iteration,
            }
            ckpt_path = os.path.join(out_dir, f"heartgpt_pretrain_iter{iteration}.pth")
            torch.save(ckpt, ckpt_path)
            log.info(f"Saved checkpoint to {ckpt_path}")

    log.info("Pretraining complete.")


if __name__ == "__main__":
    main()
