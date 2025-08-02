# scripts/generate.py

import sys
from pathlib import Path

# ensure that models input properly, without having to make the package installable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import yaml
import pandas as pd
from core.model import HeartGPTModel
from tokenise.preprocess import tokenise_biosignal


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="HeartGPT generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_type = cfg["model"]["type"]
    block_size = cfg["model"]["block_size"]
    n_embd = cfg["model"]["n_embd"]
    n_head = cfg["model"]["n_head"]
    n_layer = cfg["model"]["n_layer"]
    dropout = cfg["model"]["dropout"]
    vocab_size = cfg["model"]["vocab_size"][model_type]
    model_path = cfg["model"]["model_paths"][model_type]
    context_csv = cfg["input"]["context_csv"]
    out_model_csv = cfg["output"]["model_output"]
    out_tokenized_csv = cfg["output"]["tokenized_context"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate and load model
    model = HeartGPTModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    )
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    model = model.to(device)
    model.eval()

    # load context
    df = pd.read_csv(context_csv, header=None)
    data = df.values
    tokenized = tokenise_biosignal(data, max_length=block_size)

    tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=device)

    print("Generating new tokens...")
    generated = model.generate(tokenized_tensor, max_new_tokens=args.max_new_tokens)

    # collapse batch and convert to list
    generated_seq = generated[0].tolist()

    # save outputs
    pd.DataFrame(generated_seq).to_csv(out_model_csv, index=False, header=False)

    # save tokenized context
    tokenized_for_save = tokenized.T if tokenized.shape[0] != tokenized.shape[1] else tokenized
    pd.DataFrame(tokenized_for_save.astype(int)).to_csv(out_tokenized_csv, index=False, header=False)

    print(f"Saved generation to {out_model_csv} and context to {out_tokenized_csv}")


if __name__ == "__main__":
    main()
