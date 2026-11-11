"""Part 3 - train Model A (global context) and Model B (spatial attention).

Both variants are trained with the same seed, data order, optimiser, learning
rate, batch size and number of epochs on the same GPU. The validation loss is
computed at the end of every epoch and the weights with the lowest validation
loss are saved.

Because the ResNet-50 encoder is frozen and the image transform is deterministic,
the encoder output of an image never changes. It is therefore computed once
(``extract_features``) and re-used by both models in every epoch.
"""
import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from part1_data import SEED, FlickrSubsetDataset, prepare_data
from part2_models import EncoderCNN, ImageCaptioner, count_parameters
from part3_evaluate import bleu4_score

# BLEU-4 is exactly 0 in the first epochs (no 4-gram matches yet); silence nltk's notice.
warnings.filterwarnings("ignore", module="nltk")

OUT_DIR = Path(__file__).parent / "outputs" / "part3"
FEATURE_CACHE = Path(__file__).parent / "outputs" / "cache" / "resnet50_grid_features.pt"

# Hyper-parameters shared by both variants.
CONFIG = dict(epochs=80, batch_size=64, lr=3e-4, optimizer="Adam", grad_clip=5.0, seed=SEED)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Feature extraction pipeline (frozen ResNet-50, run once)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(data, device, batch_size=64):
    """Images -> Part 1 transform -> frozen ResNet-50 -> (N, 2048, 7, 7) grid per split."""
    if FEATURE_CACHE.exists():
        return torch.load(FEATURE_CACHE)
    encoder = EncoderCNN().to(device)
    features, start = {}, time.perf_counter()
    for name, split in data.splits.items():
        loader = DataLoader(FlickrSubsetDataset(split), batch_size=batch_size,
                            shuffle=False, num_workers=8)
        grids = []
        for images, _, _ in loader:
            grid, v = encoder(images.to(device))      # grid (B, 2048, 7, 7), v (B, 2048)
            grids.append(grid.cpu())                  # v = grid.mean((2, 3)) is recomputed later
        features[name] = torch.cat(grids)
        print(f"extracted {name}: {tuple(features[name].shape)}")
    print(f"feature extraction took {time.perf_counter() - start:.1f}s")
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, FEATURE_CACHE)
    return features


# --------------------------------------------------------------------------- #
# Training / validation
# --------------------------------------------------------------------------- #
def caption_pairs(split, device):
    """Flatten a split into (image index, caption) training pairs: 5 per image."""
    n_images, n_caps, _ = split.encoded.shape
    image_idx = torch.arange(n_images).repeat_interleave(n_caps).to(device)
    captions = split.encoded.flatten(0, 1).to(device)
    return image_idx, captions


def run_epoch(model, grid, image_idx, captions, batch_size, criterion, pad_idx,
              optimizer=None, grad_clip=None, generator=None):
    """One pass over (image, caption) pairs. Trains if an optimizer is given,
    otherwise evaluates. Returns the per-token loss and next-token accuracy."""
    training = optimizer is not None
    model.train(training)
    n = captions.size(0)
    order = (torch.randperm(n, generator=generator, device="cpu").to(captions.device)
             if training else torch.arange(n, device=captions.device))
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):
            batch = order[i:i + batch_size]
            caps = captions[batch]
            caps = caps[:, :(caps != pad_idx).sum(1).max()]   # trim padding beyond the longest caption
            targets = caps[:, 1:]                             # token to predict at each step
            logits = model(caps, grid=grid[image_idx[batch]]) # teacher forcing
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            mask = targets != pad_idx
            total_loss += loss.item() * mask.sum().item()
            total_correct += ((logits.argmax(-1) == targets) & mask).sum().item()
            total_tokens += mask.sum().item()
    return total_loss / total_tokens, total_correct / total_tokens


def train_model(variant, data, features, device, config=CONFIG, out_dir=OUT_DIR):
    """Train one variant, keep the best-validation-loss weights, return its history."""
    set_seed(config["seed"])                                  # same init / batch order for A and B
    vocab = data.vocab
    model = ImageCaptioner(len(vocab), variant, load_encoder=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    shuffle_rng = torch.Generator().manual_seed(config["seed"])

    grid = {s: features[s].to(device) for s in ("train", "val")}
    train_idx, train_caps = caption_pairs(data.splits["train"], device)
    val_idx, val_caps = caption_pairs(data.splits["val"], device)

    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc",
                               "val_bleu4", "epoch_time")}
    best_val, best_epoch, train_time = float("inf"), 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, grid["train"], train_idx, train_caps,
                                          config["batch_size"], criterion, vocab.pad_idx,
                                          optimizer, config["grad_clip"], shuffle_rng)
        val_loss, val_acc = run_epoch(model, grid["val"], val_idx, val_caps,
                                      config["batch_size"], criterion, vocab.pad_idx)
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start              # training + validation-loss pass
        train_time += epoch_time

        # Extra monitoring (not part of the timed training): greedy-decoded validation BLEU-4.
        val_bleu, _ = bleu4_score(model, grid["val"], data.splits["val"], vocab, data.max_len)

        if val_loss < best_val:                               # save the best weights
            best_val, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), out_dir / f"{variant}_best.pt")

        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc,
                                        val_bleu, epoch_time)):
            history[key].append(value)
        print(f"[{variant}] epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | val BLEU-4 {val_bleu:.4f} | "
              f"{epoch_time:.1f}s" + ("  *" if epoch == best_epoch else ""), flush=True)

    torch.save(model.state_dict(), out_dir / f"{variant}_last.pt")
    history.update(variant=variant, config=config, best_epoch=best_epoch, best_val_loss=best_val,
                   train_time_sec=train_time, learnable_params=count_parameters(model),
                   device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    with open(out_dir / f"{variant}_history.json", "w") as f:
        json.dump(history, f, indent=1)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["global", "attention"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    config = dict(CONFIG, epochs=args.epochs, lr=args.lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_data()
    features = extract_features(data, device)
    for variant in args.variants:                             # one after the other, same GPU
        h = train_model(variant, data, features, device, config, args.out)
        print(f"[{variant}] best epoch {h['best_epoch']} (val loss {h['best_val_loss']:.4f}), "
              f"training time {h['train_time_sec']:.1f}s, "
              f"learnable parameters {h['learnable_params']:,}")
