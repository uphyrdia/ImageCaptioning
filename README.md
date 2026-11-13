# Image Captioning: Global Context vs. Attention

A coursework project to get familiar with LSTM decoders and attention. Two CNN-RNN captioning models are trained on a random 2,000-image subset of Flickr8k and compared:

- **Model A (global context):** the LSTM receives the same global-average-pooled ResNet-50 vector at every step.
- **Model B (spatial attention):** the LSTM receives a new context vector at every step, built by scaled dot-product attention over the 7×7 ResNet-50 feature grid.

Both share a frozen ImageNet-pretrained ResNet-50 encoder and a single-layer LSTM decoder, and are trained with teacher forcing and decoded with greedy search.

The full write-up is in [docs/index.md](docs/index.md).

## Results

| | Model A | Model B |
|---|---|---|
| Test BLEU-4 | 0.1560 | 0.1607 |
| Training time (80 epochs) | 111.3 s | 222.8 s |
| Learnable parameters | 4,301,514 | 4,891,338 |

## Code

| File | Purpose |
|---|---|
| `dataset_acquisition.py` | Downloads Flickr8k from Kaggle |
| `part1_data.py` | Subset selection, image transform, vocabulary, tokenisation |
| `part2_models.py` | Encoder, the two decoders, main model class |
| `part3_train.py` | Trains both models and saves the best checkpoints |
| `part3_evaluate.py` | BLEU-4, loss curves, example captions, attention maps |

## Running

```bash
pip install -r requirements.txt
```

```bash
python dataset_acquisition.py
```

```bash
python part3_train.py
```

```bash
python part3_evaluate.py
```

All randomness uses seed 595. Outputs are written to `outputs/`.
