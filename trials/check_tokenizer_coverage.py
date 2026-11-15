"""
Quick diagnostic: how well does the IndicBERT v2 tokenizer actually cover
the cleaned Kannada-English comments? If a big chunk of tokens come out as
[UNK] or single-byte fallback pieces, that's a hard ceiling no amount of
epochs/architecture tweaking on the classifier head can fix -- it means the
encoder never sees usable Kannada-script signal.

Run from the same place as train_task_a.py (needs the same data/ folder
and an internet connection to pull the tokenizer).
"""
import html
import re
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"


def fix_mojibake(s):
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def clean_text(t):
    t = fix_mojibake(str(t))
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"([!?.,])\1{2,}", r"\1\1", t)
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
unk_id = tokenizer.unk_token_id

df = pd.read_csv(DATA_DIR / "binary_train.csv").dropna(subset=["Comment", "Label"])
sample = df.sample(n=min(500, len(df)), random_state=42)

total_tokens = 0
total_unk = 0
kannada_char_re = re.compile(r"[\u0C80-\u0CFF]")   # Kannada Unicode block
worst_examples = []

for raw in sample["Comment"]:
    cleaned = clean_text(raw)
    has_kannada = bool(kannada_char_re.search(cleaned))
    ids = tokenizer(cleaned, add_special_tokens=False)["input_ids"]
    n_unk = sum(1 for i in ids if i == unk_id)
    total_tokens += len(ids)
    total_unk += n_unk
    if has_kannada and len(ids) > 0:
        unk_ratio = n_unk / len(ids)
        worst_examples.append((unk_ratio, raw, cleaned, tokenizer.convert_ids_to_tokens(ids)))

worst_examples.sort(key=lambda x: -x[0])

print(f"Overall UNK rate across {len(sample)} sampled comments: {total_unk}/{total_tokens} = {100 * total_unk / max(total_tokens,1):.2f}%")
print(f"Comments containing native Kannada script in this sample: {sum(1 for _,r in zip(worst_examples, sample) if True)}")
print("\nTop 5 worst (highest UNK ratio) comments containing Kannada script:")
for unk_ratio, raw, cleaned, toks in worst_examples[:5]:
    print(f"\n  UNK ratio: {unk_ratio:.2f}")
    print(f"  raw     : {raw[:120]!r}")
    print(f"  cleaned : {cleaned[:120]!r}")
    print(f"  tokens  : {toks[:25]}")