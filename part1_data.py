"""Part 1 - Data loading and preprocessing for the Flickr8k captioning subset.

Pipeline
--------
1. Read ``captions.txt`` (8,091 images x 5 captions).
2. Draw 2,000 images with ``random.sample(image_list, 2000)`` (seed 595) and
   split them 1,600 / 200 / 200 into train / validation / test.
3. Image transform: Resize(256x256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize.
4. Text: lowercase + strip non-alphanumerics -> <start> ... <end> framing ->
   vocabulary from the training captions only (min. frequency 5) ->
   integer encoding with <unk> fallback -> <pad> to the longest training sequence.
"""
import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 595                      
SUBSET_SIZE = 2000
N_TRAIN, N_VAL, N_TEST = 1600, 200, 200
MIN_WORD_FREQ = 5               # words seen < 5 times in training become <unk>

PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN = "<pad>", "<start>", "<end>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]   # ids 0, 1, 2, 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_ROOT = Path.home() / ".cache/kagglehub/datasets/adityajn105/flickr8k/versions/1"
IMAGE_DIR = DATA_ROOT / "Images"
CAPTIONS_FILE = DATA_ROOT / "captions.txt"


# --------------------------------------------------------------------------- #
# 1. Data loading and random subset selection
# --------------------------------------------------------------------------- #
def load_captions(captions_file=CAPTIONS_FILE):
    """Read captions.txt into {image file name: [caption_1, ..., caption_5]}."""
    captions = {}
    with open(captions_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                                    # skip the "image,caption" header
        for image_name, caption in reader:
            captions.setdefault(image_name, []).append(caption)
    return captions


def select_subset(captions, seed=SEED):
    """Randomly pick 2,000 images and split them into train / val / test.

    ``random.sample`` returns the chosen images in random order, so slicing the
    sample into consecutive blocks is itself a random 1,600 / 200 / 200 split.
    """
    image_list = sorted(captions.keys())                # fixed order -> reproducible sample
    random.seed(seed)
    subset = random.sample(image_list, SUBSET_SIZE)
    return {
        "train": subset[:N_TRAIN],
        "val": subset[N_TRAIN:N_TRAIN + N_VAL],
        "test": subset[N_TRAIN + N_VAL:],
    }


# --------------------------------------------------------------------------- #
# 2. Image preprocessing (ImageNet format expected by the pre-trained ResNet-50)
# --------------------------------------------------------------------------- #
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),                      # 1. resize to 256 x 256
    transforms.CenterCrop(224),                         # 2. centre crop to 224 x 224
    transforms.ToTensor(),                              # 3. PIL image -> float tensor in [0, 1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # 4. ImageNet mean / std
])


# --------------------------------------------------------------------------- #
# 3. Text normalisation, framing, vocabulary, tokenisation and padding
# --------------------------------------------------------------------------- #
def normalize_caption(caption):
    """Lowercase, drop every non-alphanumeric character, split into words."""
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)       # remove punctuation / symbols
    return caption.split()


def frame_caption(words):
    """Prepend <start> and append <end>."""
    return [START_TOKEN] + words + [END_TOKEN]


class Vocabulary:
    """word <-> index mappings built strictly from the training captions."""

    def __init__(self, train_word_lists, min_freq=MIN_WORD_FREQ):
        self.word_counts = Counter(w for words in train_word_lists for w in words)
        kept = sorted(w for w, c in self.word_counts.items() if c >= min_freq)
        self.idx2word = SPECIAL_TOKENS + kept           # <pad>=0, <start>=1, <end>=2, <unk>=3
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.pad_idx = self.word2idx[PAD_TOKEN]
        self.start_idx = self.word2idx[START_TOKEN]
        self.end_idx = self.word2idx[END_TOKEN]
        self.unk_idx = self.word2idx[UNK_TOKEN]

    def __len__(self):
        return len(self.idx2word)

    def encode(self, framed_words):
        """Words -> integer ids; out-of-vocabulary words map to <unk>."""
        return [self.word2idx.get(w, self.unk_idx) for w in framed_words]

    def decode(self, ids, strip_special=True):
        """Integer ids -> words (optionally dropping <pad>/<start>/<end>)."""
        skip = {self.pad_idx, self.start_idx, self.end_idx} if strip_special else set()
        return [self.idx2word[i] for i in ids if i not in skip]


def pad_sequence_to(ids, max_len, vocab):
    """Right-pad with <pad> to ``max_len`` (the longest training sequence).

    A validation / test caption that is longer than every training caption is
    truncated so that it still ends with <end>.
    """
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [vocab.end_idx]
    return ids + [vocab.pad_idx] * (max_len - len(ids))


# --------------------------------------------------------------------------- #
# 4. Putting everything together
# --------------------------------------------------------------------------- #
@dataclass
class SplitData:
    image_names: list            # N image file names
    raw_captions: list           # N x 5 original caption strings
    words: list                  # N x 5 normalised word lists (no special tokens)
    encoded: torch.Tensor        # (N, 5, max_len) padded integer sequences


@dataclass
class CaptionData:
    splits: dict                 # {"train" | "val" | "test": SplitData}
    vocab: Vocabulary
    max_len: int                 # longest framed training sequence


def prepare_data(captions_file=CAPTIONS_FILE, seed=SEED):
    """Run the full Part 1 pipeline and return the three preprocessed splits."""
    captions = load_captions(captions_file)
    split_names = select_subset(captions, seed)

    # Normalise every caption of every split.
    words = {s: [[normalize_caption(c) for c in captions[n]] for n in names]
             for s, names in split_names.items()}

    # Vocabulary and maximum length come from the *training* split only.
    train_word_lists = [w for image_caps in words["train"] for w in image_caps]
    vocab = Vocabulary(train_word_lists)
    max_len = max(len(frame_caption(w)) for w in train_word_lists)

    splits = {}
    for s, names in split_names.items():
        encoded = [[pad_sequence_to(vocab.encode(frame_caption(w)), max_len, vocab)
                    for w in image_caps] for image_caps in words[s]]
        splits[s] = SplitData(
            image_names=names,
            raw_captions=[captions[n] for n in names],
            words=words[s],
            encoded=torch.tensor(encoded, dtype=torch.long),
        )
    return CaptionData(splits=splits, vocab=vocab, max_len=max_len)


class FlickrSubsetDataset(Dataset):
    """One item per image: (preprocessed image tensor, its 5 encoded captions, index)."""

    def __init__(self, split, image_dir=IMAGE_DIR, transform=image_transform):
        self.split = split
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.split.image_names)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir / self.split.image_names[idx]).convert("RGB")
        return self.transform(image), self.split.encoded[idx], idx


# --------------------------------------------------------------------------- #
# 5. Deliverable: 3 random subset images with their 5 preprocessed captions
# --------------------------------------------------------------------------- #
def denormalize(image_tensor):
    """Undo the ImageNet normalisation so a preprocessed tensor can be displayed."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def show_examples(data, out_path, n_examples=3, seed=SEED):
    """Plot ``n_examples`` random subset images next to their tokenised captions."""
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pool the three splits back into the full 2,000-image subset and sample from it.
    pool = [(s, i) for s in ("train", "val", "test")
            for i in range(len(data.splits[s].image_names))]
    chosen = random.Random(seed).sample(pool, n_examples)

    fig, axes = plt.subplots(n_examples, 2, figsize=(15, 4.7 * n_examples),
                             gridspec_kw={"width_ratios": [1, 2.6]}, facecolor="white")
    for row, (split_name, idx) in enumerate(chosen):
        split = data.splits[split_name]
        image, encoded, _ = FlickrSubsetDataset(split)[idx]

        ax_img, ax_txt = axes[row]
        ax_img.imshow(denormalize(image))
        ax_img.set_title(f"{split.image_names[idx]}  ({split_name} split)\n"
                         f"preprocessed tensor {tuple(image.shape)}", fontsize=10)
        ax_img.axis("off")
        ax_img.set_anchor("N")                           # align image with the text block

        lines = []
        for k in range(5):
            ids = encoded[k].tolist()
            tokens = [data.vocab.idx2word[i] for i in ids if i != data.vocab.pad_idx]
            lines.append(f"[{k + 1}] raw   : {split.raw_captions[idx][k]}")
            lines += textwrap.wrap(" ".join(tokens), 92, initial_indent="    tokens: ",
                                   subsequent_indent="            ")
            lines += textwrap.wrap(" ".join(map(str, ids)), 92, initial_indent="    ids   : ",
                                   subsequent_indent="            ")
            lines.append("")
        ax_txt.text(0, 1, "\n".join(lines), family="monospace", fontsize=8.6,
                    va="top", ha="left", transform=ax_txt.transAxes, color="#0b0b0b")
        ax_txt.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return chosen


if __name__ == "__main__":
    data = prepare_data()
    vocab = data.vocab
    print(f"vocabulary size : {len(vocab)} (incl. {len(SPECIAL_TOKENS)} special tokens)")
    print(f"distinct training words: {len(vocab.word_counts)}")
    print(f"max sequence length (train, incl. <start>/<end>): {data.max_len}")
    for name, split in data.splits.items():
        all_words = [w for caps in split.words for words in caps for w in words]
        n_unk = sum(w not in vocab.word2idx for w in all_words)
        longest = max(len(words) + 2 for caps in split.words for words in caps)
        print(f"{name:5s}: {len(split.image_names)} images, "
              f"{split.encoded.shape[0] * split.encoded.shape[1]} captions, "
              f"encoded {tuple(split.encoded.shape)}, longest {longest}, "
              f"<unk> rate {n_unk / len(all_words):.2%}")
    print("first 3 training images:", data.splits["train"].image_names[:3])

    out_dir = Path(__file__).parent / "outputs" / "part1"
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = show_examples(data, out_dir / "sample_images_captions.png")
    print("displayed examples:", chosen)
