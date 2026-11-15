"""
Shared helpers for the HASTIKA @ ICON-2026 pipeline.
"""
import csv
import re
import html


def find_id_key(row_keys):
    for k in row_keys:
        if (k or "").strip().lower() in ("id", "index"):
            return k
    return None


def find_comment_key(row_keys):
    for k in row_keys:
        if (k or "").strip().lower() in ("comment", "text", "tweet"):
            return k
    return None


def read_csv_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fix_mojibake(text: str) -> str:
    """
    Repairs UTF-8 text that was incorrectly decoded as Latin-1 somewhere
    upstream and re-saved -- affects ~16% of rows in this dataset, mostly
    Kannada-script segments and some emoji, e.g. 'à²à³à²¹à³à²²à²¿' should be
    'ಕೊಹ್ಲಿ'. Falls back to the original string when the round-trip isn't
    valid (i.e. the text was never mojibake to begin with).
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_text(text: str) -> str:
    """
    Light-touch cleaning for Kannada-English code-mixed YouTube comments.
    We deliberately do NOT strip Kannada script, transliterated words, or
    emojis -- they carry signal for hate/target classification. We only:
      - repair mojibake-corrupted Kannada script / emoji (see fix_mojibake)
      - unescape HTML entities (&quot; etc.) and strip stray HTML tags (<br>)
      - collapse excess whitespace
      - strip leading/trailing junk
    """
    if text is None:
        return ""
    text = fix_mojibake(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)          # drop HTML tags like <br>
    text = re.sub(r"http\S+|www\.\S+", " <url> ", text)  # normalize URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


TASK_A_LABELS = ["Non-Hate", "Hate"]
TASK_B_LABELS = [
    "Gender",
    "Geo-political",
    "Others",
    "Political",
    "Religion",
    "Violence"
]