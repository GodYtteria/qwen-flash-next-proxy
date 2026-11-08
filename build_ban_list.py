"""Build ban_ids.json: every vocab token outside English/Tagalog/code (CJK, Hangul, Cyrillic, Arabic, ...) plus emoji.

    python build_ban_list.py <tokenizer-dir> [out.json]

<tokenizer-dir> holds the model's vocab.json (GPT-2 byte-level BPE, as Qwen ships it).
"""
import json
import sys
import unicodedata
from pathlib import Path

KEEP_SCRIPTS = ("LATIN", "COMMON", "INHERITED")


def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def decode(token, dec):
    try:
        return bytes(dec[c] for c in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def is_emoji(ch):
    o = ord(ch)
    return (0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x1F1E6 <= o <= 0x1F1FF
            or 0x2300 <= o <= 0x23FF or 0x2B00 <= o <= 0x2BFF or o in (0xFE0F, 0x200D))


def foreign_script(ch):
    if ch.isspace() or ord(ch) < 0x80 or unicodedata.category(ch)[0] in "PSNZC":
        return False
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return True
    return not name.startswith(KEEP_SCRIPTS)


def banned(text):
    return any(is_emoji(c) or foreign_script(c) for c in text)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    vocab_path = Path(sys.argv[1]) / "vocab.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ban_ids.json")
    with open(vocab_path, encoding="utf-8") as fh:
        vocab = json.load(fh)
    dec = byte_decoder()
    ban = sorted(tid for tok, tid in vocab.items() if (s := decode(tok, dec)) and banned(s))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ban, fh)
    print(f"vocab {len(vocab)}  banned {len(ban)} ({len(ban) / len(vocab):.1%})  -> {out}")


if __name__ == "__main__":
    main()
