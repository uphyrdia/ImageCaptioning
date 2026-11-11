"""Part 3 - evaluation: corpus-level BLEU-4, loss curves, qualitative examples
and attention heat-maps for the two trained captioning models."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image

from part1_data import IMAGE_DIR, SEED, prepare_data
from part2_models import ImageCaptioner

OUT_DIR = Path(__file__).parent / "outputs" / "part3"
NAMES = {"global": "Model A (global context)", "attention": "Model B (spatial attention)"}
COLORS = {"global": "#2a78d6", "attention": "#eb6834"}     # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


# --------------------------------------------------------------------------- #
# Caption generation and corpus-level BLEU-4
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_captions(model, grid, vocab, max_len, batch_size=100):
    """Greedy-decode a caption for every image feature grid.

    Returns a list of word lists (cut at the first <end>) and, for Model B,
    a list of (n_words + 1, 49) attention arrays (the +1 is the <end> step).
    """
    model.eval()
    hypotheses, attention = [], []
    for i in range(0, grid.size(0), batch_size):
        ids, alphas = model.caption(vocab.start_idx, vocab.end_idx, max_len,
                                    grid=grid[i:i + batch_size])
        for b, seq in enumerate(ids.tolist()):
            length = seq.index(vocab.end_idx) if vocab.end_idx in seq else len(seq)
            hypotheses.append([vocab.idx2word[t] for t in seq[:length]])
            if alphas is not None:
                attention.append(alphas[b, :length + 1].cpu().numpy())
    return hypotheses, attention


def bleu4_score(model, grid, split, vocab, max_len):
    """Corpus-level BLEU-4 of the greedy captions against the 5 references per image.

    * references : the five normalised human captions of each image (lower-cased,
                   punctuation removed, NOT mapped to <unk>, no special tokens);
    * hypotheses : generated words up to (excluding) <end>. A generated <unk>
                   is kept as a literal token, so it can never match a reference.
    corpus_bleu pools the clipped n-gram counts (n = 1..4, uniform weights) over
    the whole split before taking the geometric mean and the brevity penalty -
    it is not an average of per-sentence BLEU scores.
    """
    hypotheses, _ = generate_captions(model, grid, vocab, max_len)
    references = split.words                         # N images x 5 reference word lists
    score = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return score, hypotheses


def paired_bootstrap(references, hyp_a, hyp_b, n_boot=1000, seed=SEED):
    """Resample test images with replacement to gauge how much of the BLEU-4
    difference between two models is test-set sampling noise."""
    rng = np.random.default_rng(seed)
    n = len(references)
    scores = np.empty((n_boot, 2))
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        refs = [references[i] for i in idx]
        scores[k, 0] = corpus_bleu(refs, [hyp_a[i] for i in idx])
        scores[k, 1] = corpus_bleu(refs, [hyp_b[i] for i in idx])
    diff = scores[:, 1] - scores[:, 0]
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    return {"ci95_a": ci(scores[:, 0]), "ci95_b": ci(scores[:, 1]),
            "ci95_diff_b_minus_a": ci(diff), "p_b_better": float((diff > 0).mean())}


# --------------------------------------------------------------------------- #
# Attention visualisation
# --------------------------------------------------------------------------- #
def load_model_view(image_name):
    """The part of the ORIGINAL image the network sees, at its native aspect ratio.

    Resize(256x256) followed by CenterCrop(224) keeps the central 224/256 = 87.5 %
    of the width and of the height, so the 7x7 attention grid maps exactly onto
    this crop of the original photo.
    """
    image = Image.open(IMAGE_DIR / image_name).convert("RGB")
    w, h = image.size
    mx, my = w * 16 / 256, h * 16 / 256
    return image.crop((round(mx), round(my), round(w - mx), round(h - my)))


def upsample_attention(alpha, size):
    """(49,) attention weights -> 7x7 grid -> smooth map of ``size`` = (width, height)."""
    grid = torch.tensor(alpha, dtype=torch.float32).view(1, 1, 7, 7)
    up = torch.nn.functional.interpolate(grid, size=(size[1], size[0]),
                                         mode="bicubic", align_corners=False)
    return up.clamp(min=0)[0, 0].numpy()


def plot_attention(image_name, words, alphas, out_path, n_cols=None):
    """One panel per generated word: the attention map alpha_t (7x7, up-sampled)
    overlaid on the original image. All panels share one colour scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = load_model_view(image_name)
    steps = words + ["<end>"]
    aspect = image.size[1] / image.size[0]
    n_cols = n_cols or (8 if aspect > 1.1 else 5)        # more columns for portrait images
    n_rows = int(np.ceil((len(steps) + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, facecolor="white",
                             figsize=(2.9 * n_cols, 2.9 * aspect * n_rows + 0.5 * n_rows + 0.6))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    axes.flat[0].imshow(image)
    axes.flat[0].set_title("input image", fontsize=11, color=INK)

    vmax = float(alphas.max())
    for t, word in enumerate(steps):
        ax = axes.flat[t + 1]
        ax.imshow(image)
        overlay = ax.imshow(upsample_attention(alphas[t], image.size), cmap="magma",
                            alpha=0.6, vmin=0, vmax=vmax)
        ax.set_title(f"{word}\nmax α = {alphas[t].max():.2f}", fontsize=11, color=INK)
    fig.suptitle("Model B: \"" + " ".join(words) + "\"", fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.93, 0.98))
    cax = fig.add_axes([0.945, 0.25, 0.012, 0.5])
    cbar = fig.colorbar(overlay, cax=cax)
    cbar.set_label("attention weight α (uniform = 0.020)", fontsize=9, color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    cbar.solids.set_alpha(1)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figures: learning curves and qualitative examples
# --------------------------------------------------------------------------- #
def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)


def plot_loss_curves(histories, out_path):
    """Training (solid) and validation (dashed) loss of both models on one graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor="white")
    for variant, h in histories.items():
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        short = NAMES[variant].split(" (")[0]
        ax.plot(epochs, h["train_loss"], color=COLORS[variant], lw=2, label=f"{short} - training")
        ax.plot(epochs, h["val_loss"], color=COLORS[variant], lw=2, ls=(0, (4, 2)),
                label=f"{short} - validation")
        best = h["best_epoch"]
        ax.scatter([best], [h["val_loss"][best - 1]], s=70, color=COLORS[variant],
                   edgecolor="white", linewidth=2, zorder=5)
        ax.annotate(f"{short} best: epoch {best}\nval loss {h['best_val_loss']:.3f}",
                    (best, h["val_loss"][best - 1]), textcoords="offset points",
                    xytext=(14, 26 if variant == "global" else -38), fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    _style(ax, "epoch", "cross-entropy loss per token",
           "Training and validation loss - Model A vs. Model B")
    ax.set_xlim(0, len(epochs) + 1)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_accuracy_bleu(histories, out_path):
    """Teacher-forced next-token accuracy and greedy validation BLEU-4 per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")
    for variant, h in histories.items():
        epochs = np.arange(1, len(h["train_acc"]) + 1)
        short = NAMES[variant].split(" (")[0]
        ax1.plot(epochs, h["train_acc"], color=COLORS[variant], lw=2, label=f"{short} - training")
        ax1.plot(epochs, h["val_acc"], color=COLORS[variant], lw=2, ls=(0, (4, 2)),
                 label=f"{short} - validation")
        ax2.plot(epochs, h["val_bleu4"], color=COLORS[variant], lw=2, label=short)
        ax2.axvline(h["best_epoch"], color=COLORS[variant], lw=1, ls=":")
    _style(ax1, "epoch", "next-token accuracy (teacher forcing)", "Token accuracy")
    _style(ax2, "epoch", "corpus BLEU-4 (greedy decoding)",
           "Validation BLEU-4 (dotted line = best-validation-loss epoch)")
    ax1.legend(frameon=False, fontsize=10, labelcolor=INK, ncols=2, loc="upper left")
    ax2.legend(frameon=False, fontsize=10, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_qualitative(split, indices, hyps, out_path):
    """Test images with their 5 ground-truth captions and both generated captions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(indices), 2, figsize=(14, 3.9 * len(indices)), facecolor="white",
                             gridspec_kw={"width_ratios": [1, 2.4]})
    for row, idx in enumerate(indices):
        ax_img, ax_txt = axes[row]
        ax_img.imshow(Image.open(IMAGE_DIR / split.image_names[idx]).convert("RGB"))
        ax_img.set_title(split.image_names[idx], fontsize=9, color=MUTED)
        ax_img.axis("off")
        ax_txt.axis("off")
        y = 0.97
        ax_txt.text(0, y, "Ground-truth captions", fontsize=11, weight="bold", color=INK,
                    va="top", transform=ax_txt.transAxes)
        for k, cap in enumerate(split.raw_captions[idx]):
            y -= 0.085
            ax_txt.text(0.01, y, f"{k + 1}. {cap}", fontsize=10.5, color=MUTED, va="top",
                        transform=ax_txt.transAxes)
        y -= 0.14
        for variant in ("global", "attention"):
            ax_txt.plot([0.0, 0.0], [y - 0.16, y], color=COLORS[variant], lw=4,
                        transform=ax_txt.transAxes, clip_on=False, solid_capstyle="butt")
            ax_txt.text(0.02, y, NAMES[variant], fontsize=11, weight="bold", color=INK,
                        va="top", transform=ax_txt.transAxes)
            ax_txt.text(0.02, y - 0.085, " ".join(hyps[variant][idx]), fontsize=11.5, color=INK,
                        va="top", transform=ax_txt.transAxes)
            y -= 0.23
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def attention_by_word(hyps, attention, top_k=14):
    """Average sharpness of the attention map used to predict each frequent word."""
    stats = {}
    for words, alphas in zip(hyps, attention):
        for word, a in zip(words + ["<end>"], alphas):
            entropy = float(-(a * np.log(a + 1e-12)).sum())
            stats.setdefault(word, []).append((float(a.max()), np.exp(entropy)))
    frequent = sorted(stats, key=lambda w: -len(stats[w]))[:top_k]
    return {w: {"count": len(stats[w]),
                "mean_max_weight": float(np.mean([m for m, _ in stats[w]])),
                "mean_effective_regions": float(np.mean([e for _, e in stats[w]]))}
            for w in frequent}


def caption_statistics(hyps, train_caption_set):
    """Simple descriptive statistics of a set of generated captions."""
    strings = [" ".join(h) for h in hyps]
    return {
        "mean_length": float(np.mean([len(h) for h in hyps])),
        "distinct_captions": len(set(strings)),
        "distinct_words": len({w for h in hyps for w in h}),
        "unk_tokens": sum(w == "<unk>" for h in hyps for w in h),
        "copied_from_training_set": sum(s in train_caption_set for s in strings),
    }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_data()
    vocab, test, val = data.vocab, data.splits["test"], data.splits["val"]
    features = torch.load(Path(__file__).parent / "outputs/cache/resnet50_grid_features.pt")
    test_grid, val_grid = features["test"].to(device), features["val"].to(device)
    train_caption_set = {" ".join(w) for caps in data.splits["train"].words for w in caps}

    histories, metrics, hyps, attention = {}, {}, {}, None
    for variant in ("global", "attention"):
        with open(OUT_DIR / f"{variant}_history.json") as f:
            histories[variant] = h = json.load(f)
        model = ImageCaptioner(len(vocab), variant, load_encoder=False).to(device)

        # Final model = weights with the lowest validation loss.
        model.load_state_dict(torch.load(OUT_DIR / f"{variant}_best.pt"))
        hyps[variant], att = generate_captions(model, test_grid, vocab, data.max_len)
        bleu = {f"bleu{n}": corpus_bleu(test.words, hyps[variant], weights=[1 / n] * n)
                for n in (1, 2, 3, 4)}
        assert abs(bleu["bleu4"] - bleu4_score(model, test_grid, test, vocab, data.max_len)[0]) < 1e-12
        val_bleu_best = bleu4_score(model, val_grid, val, vocab, data.max_len)[0]
        if variant == "attention":
            attention = att
            a = np.concatenate(att)                                   # all decoding steps x 49
            entropy = -(a * np.log(a + 1e-12)).sum(1)
            metrics["attention_stats"] = {
                "mean_entropy_nats": float(entropy.mean()), "uniform_entropy_nats": float(np.log(49)),
                "mean_max_weight": float(a.max(1).mean()), "uniform_weight": 1 / 49,
                "mean_effective_regions": float(np.exp(entropy).mean())}
            metrics["attention_by_word"] = attention_by_word(hyps[variant], att)

        # For reference only: the weights after the last epoch.
        model.load_state_dict(torch.load(OUT_DIR / f"{variant}_last.pt"))
        last_test = bleu4_score(model, test_grid, test, vocab, data.max_len)[0]
        last_val = bleu4_score(model, val_grid, val, vocab, data.max_len)[0]

        metrics[variant] = {
            "test": bleu, "best_epoch": h["best_epoch"], "best_val_loss": h["best_val_loss"],
            "val_bleu4_best_ckpt": val_bleu_best,
            "last_epoch_ckpt": {"test_bleu4": last_test, "val_bleu4": last_val,
                                "val_loss": h["val_loss"][-1], "train_loss": h["train_loss"][-1]},
            "train_loss_at_best": h["train_loss"][h["best_epoch"] - 1],
            "train_time_sec": h["train_time_sec"],
            "mean_epoch_time_sec": float(np.mean(h["epoch_time"])),
            "learnable_params": h["learnable_params"],
            "caption_stats": caption_statistics(hyps[variant], train_caption_set),
        }
        print(f"{NAMES[variant]}: test BLEU-4 = {bleu['bleu4']:.4f} | time "
              f"{h['train_time_sec']:.1f}s | params {h['learnable_params']:,}")

    metrics["bootstrap"] = paired_bootstrap(test.words, hyps["global"], hyps["attention"])
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    with open(OUT_DIR / "test_captions.json", "w") as f:
        json.dump([{"image": test.image_names[i], "references": test.raw_captions[i],
                    "model_a": " ".join(hyps["global"][i]),
                    "model_b": " ".join(hyps["attention"][i])}
                   for i in range(len(test.image_names))], f, indent=1)
    print(json.dumps(metrics, indent=1))

    plot_loss_curves(histories, OUT_DIR / "loss_curves.png")
    plot_accuracy_bleu(histories, OUT_DIR / "accuracy_bleu_curves.png")

    # Three random test images (seed 595) for the qualitative comparison.
    chosen = random.Random(SEED).sample(range(len(test.image_names)), 3)
    plot_qualitative(test, chosen, hyps, OUT_DIR / "qualitative_examples.png")
    for k, idx in enumerate(chosen):
        plot_attention(test.image_names[idx], hyps["attention"][idx], attention[idx],
                       OUT_DIR / f"attention_map_{k + 1}.png")
    print("qualitative test indices:", chosen)
