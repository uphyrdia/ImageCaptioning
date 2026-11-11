# Project 4: Image Captioning with CNN-RNN Architectures — Global Context vs. Attention

## AI Usage Statement

**AI tool used: Claude Code (Anthropic), running the Claude Fable 5.2 model.**
The tool was used to write the Python code, run the experiments on my GPU, produce the figures, and draft the text of this report. All numbers, tables and figures below come from running the code in this folder with seed 595.

---

# Part 1: Data Loading and Preprocessing

The constants used throughout Part 1:

![Configuration constants](outputs/part1/Configuration.png)

## 1.1 Data loading and random subset selection

Flickr8k was downloaded from Kaggle (`adityajn105/flickr8k`). `select_subset` seeds the RNG with **595**, draws 2,000 images with `random.sample(image_list, 2000)`, and splits them into 1,600 training, 200 validation and 200 test images together with their captions.

![Data loading and splitting code](outputs/part1/data_loading_splitting.png)

## 1.2 Image preprocessing

![Image transform code](outputs/part1/image_transform.png)

## 1.3 Text normalisation and tokenisation

1. **Normalisation** — lower-case and remove all non-alphanumeric characters (`normalize_caption`).
2. **Framing** — prepend `<start>` and append `<end>` (`frame_caption`).
3. **Vocabulary** — built from the training captions only, keeping words that appear at least 5 times, plus `<pad>`, `<start>`, `<end>`, `<unk>`; `word2idx` and `idx2word` are the two mappings (`Vocabulary`).
4. **Tokenisation and padding** — words are mapped to integers, unknown words to `<unk>`, and every sequence is padded with `<pad>` to the longest training sequence (`Vocabulary.encode`, `pad_sequence_to`).

![Text normalisation, vocabulary and padding code](outputs/part1/text_normalization_vocabulary.png)

`prepare_data` applies these steps to the three splits. The vocabulary and the padding length are computed from the training split only.

![Pipeline assembly code](outputs/part1/prepare_data.png)

`FlickrSubsetDataset` returns each image after the image transform, together with its 5 encoded captions.

![Dataset class code](outputs/part1/FlickrSubsetDataset.png)

The resulting vocabulary has **1,226** entries (1,222 words + 4 special tokens) and every caption is padded to length **35**.

## 1.4 Visual display: 3 random subset images with their preprocessed captions

![Three random subset images with tokenised captions](outputs/part1/sample_images_captions.png)

---

# Part 2: Architectural Implementation

## 2.1 Freeze or fine-tune the ResNet-50 encoder?

**I froze the ResNet-50 encoder** for both Model A and Model B.

- **Overfitting risk.** The training set has only 1,600 images, while the encoder has 23.5 M parameters. Fine-tuning it would mostly memorise the training images.
- **Fair comparison.** Both models receive exactly the same visual features, so any difference in results comes from the decoder's context formulation.
- **Efficiency.** A frozen encoder gives the same output for an image every time, so the features are extracted once and reused in every epoch of both models.

## 2.2 Feature extraction pipeline

`EncoderCNN` passes the image through the ResNet-50 stem and stages 1–4, and returns the spatial grid `(Batch, 2048, 7, 7)` and the GAP vector **v** `(Batch, 2048)`.

![Encoder code](outputs/code/p2_encoder.png)

`extract_features` runs the Part 1 transform and the frozen encoder over each split and stores the grid.

![Feature extraction code](outputs/code/p2_extract_features.png)

## 2.3 The two modular decoders

Both decoders share one base class: a 256-d embedding layer, a single-layer LSTM with a 256-d hidden state whose input is the concatenation of the embedded token and the context vector, and a fully connected classification head. They differ only in `context()`.

![Shared decoder code](outputs/code/p2_decoder_shared.png)

- **Model A (`GlobalContextDecoder`)** uses the static vector **v** at every time step.
- **Model B (`AttentionDecoder`)** computes a dynamic context $u_t$ at every step with scaled dot-product attention between the previous hidden state $h_{t-1}$ and the 49 spatial features $a_i$:

$$e_{t,i} = \frac{(W_q h_{t-1})^\top (W_k a_i)}{\sqrt{d}}, \qquad \alpha_t = \mathrm{softmax}(e_t), \qquad u_t = \sum_{i=1}^{49} \alpha_{t,i}\, a_i$$

![Model A and Model B decoder code](outputs/code/p2_decoder_variants.png)

## 2.4 Teacher forcing and autoregressive generation

**Training (teacher forcing).** At step *t* the decoder input is the ground-truth token `captions[:, t]`, and the output is scored against `captions[:, t+1]`.

![Teacher forcing code](outputs/code/p2_teacher_forcing.png)

**Inference (greedy search).** Decoding starts from `<start>`. At each step the highest-probability word is selected with `argmax` and fed back as the next input, until `<end>` is produced.

![Greedy search code](outputs/code/p2_greedy_search.png)

## 2.5 Main model class

`ImageCaptioner(vocab_size, variant)` builds Model A for `variant="global"` and Model B for `variant="attention"`; the encoder and the training and inference interfaces are shared.

![Main model class](outputs/code/p2_main_model.png)

---

# Part 3: Ablation Study and Comparative Analysis

Both variants were trained with the same settings on the same GPU (RTX 4080 SUPER).

| Setting | Value |
|---|---|
| Epochs | 80 |
| Batch size | 64 |
| Optimiser | Adam, learning rate 3 × 10⁻⁴ |
| Loss | Cross-entropy, `<pad>` ignored |
| Model selection | Validation loss after every epoch; the weights with the lowest value are saved |

## A. Empirical Results

### A.1 Loss curves

![Training and validation loss of both models](outputs/part3/loss_curves.png)

![Token accuracy and validation BLEU-4](outputs/part3/accuracy_bleu_curves.png)

### A.2 Performance metrics

| Metric | Model A (global context) | Model B (spatial attention) |
|---|---|---|
| Corpus-level BLEU-4 on the test set | **0.1560** | **0.1607** |
| Total training time | **111.3 s** | **222.8 s** |
| Learnable parameters | **4,301,514** | **4,891,338** |
| Best validation loss (epoch) | 2.880 (epoch 49) | 2.963 (epoch 35) |

BLEU-4 is computed with the saved best-validation-loss weights and greedy decoding.

### A.3 Qualitative examples and attention heat-maps

![Three test images with ground-truth and generated captions](outputs/part3/qualitative_examples.png)

Attention maps of Model B for the first image. Each panel shows the weights $\alpha_t$ used to predict that word, overlaid on the original image.

![Attention maps for "a dog is jumping over a tree"](outputs/part3/attention_map_1.png)

### A.4 Code for BLEU-4 and the attention visualisation

![BLEU-4 code](outputs/code/p3_bleu.png)

![Attention visualisation code](outputs/code/p3_attention_vis.png)

## B. Comparative Analysis

### B.1 Architectural influence

**Validation loss.** Replacing the fixed vector **v** with a dynamic context helped early and hurt later. Model B had the lower validation loss in each of the first 18 epochs (epoch 10: 3.242 vs. 3.282). The curves cross at epoch 19; after that Model A keeps improving to **2.880** at epoch 49, while Model B bottoms out at **2.963** at epoch 35 and then climbs back to 3.109 by epoch 80.

**Descriptive accuracy.** Model B's test BLEU-4 is slightly higher (0.1607 vs. 0.1560). The gap is small for a 200-image test set: resampling the test images gives a 95 % interval for the difference of −0.024 to +0.030, so the two models are effectively tied on BLEU-4. The examples show the difference in character:

- **Dog.** Model A gives the generic "a dog runs through the grass". Model B gives "a dog is jumping over a tree", which matches the references.
- **Boy.** Model A repeats itself ("a red shirt and a red shirt and a ball in a ball"). Model B is fluent but invents "white uniform" and misses the ball.
- **Girls on the beach.** Model A is vague but not wrong ("a group of people"). Model B is more specific and wrong ("two boys").

Attention made the captions more specific, which helps when the detail is right and hurts when it is not.

**Why.** Model B's context is built from a few individual grid cells, so it varies more between images than the average of all 49 cells. With only 1,600 training images the decoder used this extra freedom partly to fit the training set: by epoch 80 its training loss is 1.83 against 2.06 for Model A, yet its validation loss is higher.

### B.2 Learning dynamics

**Convergence speed.** Model B converged faster. Its training loss is below Model A's in every epoch, and it reached its best validation loss at epoch 35 against epoch 49 for Model A. In wall-clock time the advantage disappears, because a Model B epoch takes about twice as long.

**Smoothness.** Both loss curves are smooth, with no spikes in either run. Validation BLEU-4 is noisy for both models and moves by 0.01–0.02 from one epoch to the next.

**Overfitting.** Both models overfit, and Model B overfits earlier and harder. Neither is underfitting at its saved checkpoint.

| | Model A | Model B |
|---|---|---|
| Training loss falls below validation loss at | epoch 15 | epoch 12 |
| Train – validation loss gap at epoch 80 | 0.86 | 1.28 |
| Rise in validation loss from its minimum to epoch 80 | +0.032 | +0.146 |
| Training / validation token accuracy at epoch 80 | 48.1 % / 37.1 % | 52.5 % / 35.2 % |

Model A's validation accuracy is still creeping up late in training, while Model B's peaked at 35.9 % at epoch 38 and then drifted down as its training accuracy kept rising.

### B.3 Overall synthesis

For this task, with 1,600 training images and a frozen encoder, **the global-context baseline was the more effective choice**. Model A reached the lower validation loss (2.880 vs. 2.963), trained in half the time (111 s vs. 223 s), has 590 k fewer parameters, and its BLEU-4 is effectively tied with Model B's.

The attention mechanism did learn real word-to-region alignment. In the heat-map, the word *dog* puts 84 % of its weight on the dog, and the attention moves to the log for *over a tree*. Across the test set, attention is sharpest for object words such as *man* and *dog* (mean peak weight about 0.45) and most diffuse for function words such as *the* (0.17). This grounding made Model B's captions more specific, but on this small dataset it also made the model overfit sooner, so it did not produce a measurable gain in BLEU-4.

Saving the best-validation-loss weights also mattered: the epoch-80 weights score a lower test BLEU-4 for both models (0.139 for A and 0.151 for B).
