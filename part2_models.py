"""Part 2 - CNN-RNN image captioning models.

* ``EncoderCNN``           - frozen, ImageNet pre-trained ResNet-50 feature extractor.
* ``GlobalContextDecoder`` - Model A: the static GAP vector v is the context at every step.
* ``AttentionDecoder``     - Model B: a dynamic context u_t from scaled dot-product attention
                             between the 49 spatial features and the previous hidden state.
* ``ImageCaptioner``       - main model; ``variant="global"`` or ``"attention"`` swaps the decoder.
"""
import math

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

EMBED_DIM = 256       # word-embedding dimension
HIDDEN_DIM = 256      # LSTM hidden-state dimension
FEATURE_DIM = 2048    # ResNet-50 stage-4 channels
ATTENTION_DIM = 256   # dimension of the query / key space of Model B
DROPOUT = 0.5


# --------------------------------------------------------------------------- #
# CNN encoder (visual feature extraction)
# --------------------------------------------------------------------------- #
class EncoderCNN(nn.Module):
    """Pre-trained ResNet-50 without its classification head.

    Returns both visual representations required by the two decoders:
      * spatial grid, straight after stage 4 : (B, 2048, 7, 7)
      * global average pooled vector v       : (B, 2048)
    The encoder is frozen: its weights never receive gradients and it always
    runs in eval mode, so the BatchNorm statistics stay at their ImageNet values.
    """

    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stages = nn.Sequential(resnet.layer1, resnet.layer2,      # stages 1-2
                                    resnet.layer3, resnet.layer4)      # stages 3-4
        self.gap = nn.AdaptiveAvgPool2d(1)                             # global average pooling
        for p in self.parameters():                                    # freeze every weight
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        """Keep the frozen backbone in eval mode even when the captioner trains."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):                        # images: (B, 3, 224, 224)
        grid = self.stages(self.stem(images))         # (B, 2048, 7, 7) spatial grid
        v = self.gap(grid).flatten(1)                 # (B, 2048)       global context
        return grid, v


# --------------------------------------------------------------------------- #
# LSTM decoders
# --------------------------------------------------------------------------- #
class DecoderBase(nn.Module):
    """Everything the two variants share: embedding, LSTM cell, FC head and
    the teacher-forcing / greedy-search loops. Sub-classes only implement
    ``context()``, i.e. how the visual context of one time step is formed."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 feature_dim=FEATURE_DIM, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Single-layer LSTM, unrolled one step at a time. Its input is the
        # concatenation [embedded token ; context vector].
        self.lstm = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.init_h = nn.Linear(feature_dim, hidden_dim)   # h_0 from the image
        self.init_c = nn.Linear(feature_dim, hidden_dim)   # c_0 from the image
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)        # classification head

    # -- to be provided by Model A / Model B --------------------------------- #
    def context(self, regions, v, h_prev):
        """Return (context vector (B, 2048), attention weights (B, 49) or None)."""
        raise NotImplementedError

    # -- shared helpers ------------------------------------------------------ #
    @staticmethod
    def prepare_features(grid):
        """(B, 2048, 7, 7) grid -> 49 region vectors (B, 49, 2048) and GAP vector v (B, 2048)."""
        regions = grid.flatten(2).transpose(1, 2)
        return regions, regions.mean(dim=1)

    def init_state(self, v):
        """Initialise the LSTM state from the global image vector."""
        return torch.tanh(self.init_h(v)), torch.tanh(self.init_c(v))

    def step(self, tokens, regions, v, h, c):
        """One decoding step: token y_{t-1} and state (h_{t-1}, c_{t-1}) -> logits for y_t."""
        ctx, alpha = self.context(regions, v, h)                  # uses h_{t-1}
        lstm_in = torch.cat([self.dropout(self.embedding(tokens)), ctx], dim=1)
        h, c = self.lstm(lstm_in, (h, c))
        logits = self.fc(self.dropout(h))
        return logits, h, c, alpha

    # -- training: teacher forcing ------------------------------------------- #
    def forward(self, grid, captions):
        """Teacher forcing. captions: (B, T) = <start> w_1 ... w_n <end> <pad>...

        At step t the decoder input is the GROUND-TRUTH token captions[:, t]
        (never its own prediction) and the output is scored against captions[:, t+1].
        Returns logits of shape (B, T-1, vocab).
        """
        regions, v = self.prepare_features(grid)
        h, c = self.init_state(v)
        logits = []
        for t in range(captions.size(1) - 1):
            step_logits, h, c, _ = self.step(captions[:, t], regions, v, h, c)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    # -- inference: autoregressive greedy search ----------------------------- #
    @torch.no_grad()
    def generate(self, grid, start_idx, end_idx, max_len):
        """Greedy search. The input at step t is the model's OWN prediction from
        step t-1; decoding of a caption stops once it has produced <end>.

        Returns the generated ids (B, L) - positions after <end> are filled with
        <end> - and, for Model B, the attention maps (B, L, 49).
        """
        regions, v = self.prepare_features(grid)
        h, c = self.init_state(v)
        batch = grid.size(0)
        tokens = torch.full((batch,), start_idx, dtype=torch.long, device=grid.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=grid.device)
        outputs, alphas = [], []
        for _ in range(max_len - 1):                              # <start> is already given
            logits, h, c, alpha = self.step(tokens, regions, v, h, c)
            tokens = logits.argmax(dim=1)                         # highest-probability word
            tokens = tokens.masked_fill(finished, end_idx)        # freeze finished captions
            outputs.append(tokens)
            if alpha is not None:
                alphas.append(alpha)
            finished = finished | (tokens == end_idx)
            if finished.all():                                    # every caption hit <end>
                break
        return torch.stack(outputs, dim=1), (torch.stack(alphas, dim=1) if alphas else None)


class GlobalContextDecoder(DecoderBase):
    """Model A - global context baseline: the same GAP vector v at every time step."""

    def context(self, regions, v, h_prev):
        return v, None


class AttentionDecoder(DecoderBase):
    """Model B - spatial attention: a new context vector u_t at every time step.

    Scaled dot-product attention with query h_{t-1} and the 49 region vectors a_i
    as keys and values:
        e_ti    = (W_q h_{t-1}) . (W_k a_i) / sqrt(d)
        alpha_t = softmax_i(e_ti)
        u_t     = sum_i alpha_ti * a_i
    With uniform weights (alpha_ti = 1/49) u_t equals v, i.e. Model A is the
    special case of Model B that ignores the query.
    """

    def __init__(self, vocab_size, attention_dim=ATTENTION_DIM, **kwargs):
        super().__init__(vocab_size, **kwargs)
        feature_dim = self.init_h.in_features
        # Created after the shared layers so that, under the same seed, the
        # shared layers of Model A and Model B start from identical weights.
        self.query_proj = nn.Linear(self.hidden_dim, attention_dim, bias=False)   # W_q
        self.key_proj = nn.Linear(feature_dim, attention_dim, bias=False)         # W_k
        self.scale = math.sqrt(attention_dim)
        self._keys = None

    def prepare_features(self, grid):
        regions, v = super().prepare_features(grid)
        self._keys = self.key_proj(regions)            # keys do not depend on t: project once
        return regions, v

    def context(self, regions, v, h_prev):
        query = self.query_proj(h_prev).unsqueeze(2)                    # (B, d, 1)
        scores = torch.bmm(self._keys, query).squeeze(2) / self.scale   # (B, 49) alignment scores
        alpha = torch.softmax(scores, dim=1)                            # (B, 49) attention weights
        u_t = torch.bmm(alpha.unsqueeze(1), regions).squeeze(1)         # (B, 2048) weighted sum
        return u_t, alpha


# --------------------------------------------------------------------------- #
# Main model: swaps between the two variants
# --------------------------------------------------------------------------- #
DECODERS = {"global": GlobalContextDecoder,      # Model A
            "attention": AttentionDecoder}       # Model B


class ImageCaptioner(nn.Module):
    """ResNet-50 encoder + LSTM decoder. ``variant`` selects Model A or Model B;
    everything else (encoder, embedding, LSTM cell, FC head, decoding loops) is shared."""

    def __init__(self, vocab_size, variant="global", load_encoder=True):
        super().__init__()
        if variant not in DECODERS:
            raise ValueError(f"variant must be one of {list(DECODERS)}, got {variant!r}")
        self.variant = variant
        # The frozen encoder can be skipped when its outputs are served from a cache.
        self.encoder = EncoderCNN() if load_encoder else None
        self.decoder = DECODERS[variant](vocab_size)

    def encode(self, images):
        """Images (B, 3, 224, 224) -> spatial grid (B, 2048, 7, 7)."""
        grid, _ = self.encoder(images)
        return grid

    def forward(self, captions, images=None, grid=None):
        """Teacher-forced logits. Pass raw ``images`` or a pre-computed feature ``grid``."""
        if grid is None:
            grid = self.encode(images)
        return self.decoder(grid, captions)

    @torch.no_grad()
    def caption(self, start_idx, end_idx, max_len, images=None, grid=None):
        """Greedy autoregressive captioning from raw ``images`` or a cached ``grid``."""
        if grid is None:
            grid = self.encode(images)
        return self.decoder.generate(grid, start_idx, end_idx, max_len)


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


if __name__ == "__main__":
    # Smoke test: shapes, parameter counts and the "uniform attention == Model A" property.
    torch.manual_seed(595)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size, max_len = 1226, 35
    images = torch.randn(4, 3, 224, 224, device=device)
    captions = torch.randint(4, vocab_size, (4, max_len), device=device)

    encoder = EncoderCNN().to(device)
    grid, v = encoder(images)
    print("spatial grid:", tuple(grid.shape), "| GAP vector v:", tuple(v.shape))
    print("encoder parameters (frozen):", count_parameters(encoder, trainable_only=False))

    for variant in DECODERS:
        torch.manual_seed(595)
        model = ImageCaptioner(vocab_size, variant).to(device)
        logits = model(captions, images=images)
        ids, alphas = model.eval().caption(1, 2, max_len, images=images)
        print(f"{variant:9s} | logits {tuple(logits.shape)} | generated {tuple(ids.shape)} | "
              f"attention {None if alphas is None else tuple(alphas.shape)} | "
              f"learnable params {count_parameters(model):,}")
