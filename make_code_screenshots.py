"""Render the code snippets shown in report.md as PNG "screenshots".

Each snippet is cut out of the real source file (from the first line starting
with ``start`` up to, but excluding, the first later line starting with ``end``),
so the images in the report always match the code that produced the results.
"""
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "code"
FONT = "DejaVu Sans Mono"
FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# name: (file, start prefix, end prefix or None for end-of-file)
SNIPPETS = {
    # ---- Part 1 ----
    "p1_loading_split": ("part1_data.py", "import csv", "# 2. Image"),
    "p1_image_transform": ("part1_data.py", "image_transform = ", "# 3. Text"),
    "p1_text_pipeline": ("part1_data.py", "def normalize_caption", "# 4. Putting"),
    "p1_prepare_data": ("part1_data.py", "@dataclass\nclass SplitData", "# 5. Deliverable"),
    "p1_display": ("part1_data.py", "def denormalize", 'if __name__ == "__main__":'),
    # ---- Part 2 ----
    "p2_encoder": ("part2_models.py", "import math", "# LSTM decoders"),
    "p2_extract_features": ("part3_train.py", "@torch.no_grad()\ndef extract_features", "# Training / validation"),
    "p2_decoder_shared": ("part2_models.py", "class DecoderBase", "    # -- training: teacher forcing"),
    "p2_decoder_variants": ("part2_models.py", "class GlobalContextDecoder", "# Main model"),
    "p2_teacher_forcing": ("part2_models.py", "    # -- training: teacher forcing", "    # -- inference: autoregressive"),
    "p2_greedy_search": ("part2_models.py", "    # -- inference: autoregressive", "class GlobalContextDecoder"),
    "p2_main_model": ("part2_models.py", "DECODERS = ", 'if __name__ == "__main__":'),
    # ---- Part 3 ----
    "p3_run_epoch": ("part3_train.py", "def caption_pairs", "def train_model"),
    "p3_train_model": ("part3_train.py", "def train_model", 'if __name__ == "__main__":'),
    "p3_bleu": ("part3_evaluate.py", "@torch.no_grad()\ndef generate_captions", "def paired_bootstrap"),
    "p3_attention_vis": ("part3_evaluate.py", "def load_model_view", "# Figures"),
}


def cut(source, start, end):
    """Return (snippet text, 1-based first line number)."""
    i = 0 if source.startswith(start) else source.index("\n" + start) + 1
    j = source.index("\n" + end, i) + 1 if end else len(source)
    lines = source[i:j].splitlines()
    while lines and (not lines[-1].strip() or set(lines[-1].strip()) <= set("#- ")):
        lines.pop()                      # drop trailing blank lines / section rulers
    return "\n".join(lines) + "\n", source.count("\n", 0, i) + 1


def render(name, file, start, end):
    snippet, first_line = cut((ROOT / file).read_text(), start, end)
    n_lines = snippet.count("\n")
    png = highlight(snippet, PythonLexer(), ImageFormatter(
        font_name=FONT, font_size=17, line_numbers=True, line_number_start=first_line,
        style="default", line_pad=5, image_pad=14, line_number_bg="#f4f4f1",
        line_number_fg="#898781", line_number_separator=False, line_number_pad=10))
    code = Image.open(io.BytesIO(png)).convert("RGB")

    # Title bar: file name and line range.
    bar_h = 40
    width = max(code.width, 760)
    canvas = Image.new("RGB", (width, code.height + bar_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, bar_h], fill="#2c2c2a")
    draw.text((14, 9), f"{file}  ·  lines {first_line}-{first_line + n_lines - 1}",
              font=ImageFont.truetype(FONT_FILE, 18), fill="#f0efec")
    canvas.paste(code, (0, bar_h))
    ImageDraw.Draw(canvas).rectangle([0, 0, width - 1, canvas.height - 1], outline="#c3c2b7")
    canvas.save(OUT / f"{name}.png")
    return first_line, n_lines, canvas.size


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (file, start, end) in SNIPPETS.items():
        first, n, size = render(name, file, start, end)
        print(f"{name:22s} {file:20s} lines {first:3d}-{first + n - 1:3d}  {size[0]}x{size[1]}")
