# scripts/finetune.py

import sys
from pathlib import Path

# allow local imports without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import yaml
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.model import HeartGPTModel           # base transformer 
from data.labeled_sequence import LabeledSequenceDataset  # dataset, random so that it runs


# define the new head  - rather than n_embd to n_vocab, we want n_emdb to 1, for a classification task.
class NewHead(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        # feature extraction, patterns going from 64 dim to 1
        self.linear1 = nn.Sequential(nn.Linear(n_embd,1))
        self.SigM1 = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.SigM1(x).squeeze(-1)

        return x
        

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
            logging.FileHandler(os.path.join(output_dir, "finetune.log")),
        ],
    )
    return logging.getLogger("finetune")


def freeze_base_model(base_model, cfg):
    # freeze everything first
    for param in base_model.parameters():
        param.requires_grad = False
    # unfreeze last block if requested
    if cfg["freeze"].get("unfreeze_last_block", True):
        for param in base_model.blocks[-1].parameters():
            param.requires_grad = True
    # unfreeze layer norm
    if cfg["freeze"].get("unfreeze_layer_norm", True):
        for param in base_model.ln_f.parameters():
            param.requires_grad = True


def encode_with_base(base_model, idx):
    """
    Returns hidden states before lm_head: (B, T, n_embd)
    """
    B, T = idx.shape
    device = idx.device
    tok_emb = base_model.token_embedding_table(idx)
    pos_emb = base_model.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = base_model.blocks(x)
    x = base_model.ln_f(x)
    return x  # (B, T, C)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HeartGPT with NewHead (binary classification)")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--override-pretrained", type=str, default="", help="Override pretrained model path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["output"]["dir"]
    log = setup_logger(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- unpack config ---
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    freeze_cfg = cfg.get("freeze", {})
    data_cfg = cfg["data"]

    # --- base model ---
    base_model = HeartGPTModel(
        vocab_size=model_cfg["vocab_size"],
        n_embd=model_cfg["n_embd"],
        n_head=model_cfg["n_head"],
        n_layer=model_cfg["n_layer"],
        block_size=model_cfg["block_size"],
        dropout=model_cfg["dropout"],
    )
    pretrained_path = args.override_pretrained or model_cfg["pretrained_path"]
    state = torch.load(pretrained_path, map_location=device)
    base_model.load_state_dict(state)
    base_model = base_model.to(device)

    # apply freezing policy
    freeze_base_model(base_model, cfg)

    # new head
    new_head = NewHead(model_cfg["n_embd"]).to(device)

    # dataset
    train_ds = LabeledSequenceDataset(
        data_cfg["train_csv"],
        block_size=model_cfg["block_size"],
        vocab_size=model_cfg["vocab_size"],
    )
    val_ds = LabeledSequenceDataset(
        data_cfg["val_csv"],
        block_size=model_cfg["block_size"],
        vocab_size=model_cfg["vocab_size"],
    )
    train_loader = DataLoader(train_ds, batch_size=training_cfg["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"], pin_memory=True)

    try:
        learning_rate = float(training_cfg.get("learning_rate", 3e-4))
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid optimizer config: {training_cfg}") from e


    # optimizer: only params with requires_grad=True (new head + unfrozen parts)
    trainable_params = list(new_head.parameters()) + [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # loss: now binary cross entropy for a classification task
    criterion = nn.BCELoss()

    log.info(f"Starting fine-tuning. Config: {cfg}")
    log.info(f"Trainable params: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")

    # training loop
    for epoch in range(training_cfg["max_epochs"]):
        base_model.train()
        new_head.train()
        total_loss = 0.0
        for idx_batch, label_batch in train_loader:
            idx_batch = idx_batch.to(device)
            label_batch = label_batch.to(device)

            hidden = encode_with_base(base_model, idx_batch)  # (B, T, C)
            preds = new_head(hidden)  # (B,)
            loss = criterion(preds, label_batch)

            optimizer.zero_grad(set_to_none=True)
            loss = loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        log.info(f"[Epoch {epoch+1}] train loss: {avg_train_loss:.4f}")

        # validation
        base_model.eval()
        new_head.eval()
        with torch.no_grad():
            val_loss = 0.0
            for idx_batch, label_batch in val_loader:
                idx_batch = idx_batch.to(device)
                label_batch = label_batch.to(device)
                hidden = encode_with_base(base_model, idx_batch)
                preds = new_head(hidden)
                val_loss += criterion(preds, label_batch).item()
            avg_val_loss = val_loss / len(val_loader)
        log.info(f"[Epoch {epoch+1}] val loss: {avg_val_loss:.4f}")

        # checkpoint
        checkpoint = {
            "base_model": base_model.state_dict(),
            "new_head": new_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": cfg,
        }
        torch.save(checkpoint, os.path.join(output_dir, f"finetune_epoch{epoch+1}.pth"))

    log.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
